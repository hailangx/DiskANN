/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach E: Unlimited-effort paged post-filter.
//!
//! Removes the max_effort cap from the paged filter loop. Instead of stopping
//! after evaluating FILTER-EF candidates, the loop continues until either:
//!   - k matching results are found, OR
//!   - the graph returns 0 candidates (frontier exhausted)
//!
//! This mimics Redis's dual-queue approach where non-matching candidates don't
//! consume the results budget. The graph explores outward from the query point,
//! and the filter is applied purely as a post-processing step.
//!
//! Uses FullPrecision (no BetaFilter) — zero FFI callback overhead during
//! graph traversal. The only FFI calls happen in the post-filter loop.

use crate::{
    FilterCandidateCallback,
    SearchResults,
    garnet::{Context, GarnetId},
    labels::GarnetQueryLabelProvider,
    provider::{self, GarnetProvider},
};
use diskann::{
    ANNError, ANNResult,
    graph::{InplaceDeleteMethod, SearchOutputBuffer, glue::SearchStrategy, index::{SearchState, SearchStats}, search},
    neighbor::Neighbor,
    provider::{Accessor, DataProvider},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};
use std::cmp::max;
use std::sync::Arc;

/// Type-erased version of `DiskANNIndex<GarnetProvider>`.
/// All vector data is passed as untyped byte slices.
pub trait DynIndex: Send + Sync {
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()>;

    fn get_attributes(&self, context: &Context, id: &GarnetId) -> Option<Vec<u8>>;

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_vector_filtered(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn search_element_filtered(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats>;

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()>;

    fn approximate_count(&self) -> u64;

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()>;

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool;

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool;
}

impl<T: VectorRepr> DynIndex for DiskANNIndex<GarnetProvider<T>> {
    fn insert(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.insert(
            FullPrecision,
            context,
            id,
            bytemuck::cast_slice::<u8, T>(data),
        )
    }

    fn set_attributes(&self, context: &Context, id: &GarnetId, data: &[u8]) -> ANNResult<()> {
        self.inner
            .provider()
            .set_attributes(context, id, data)
            .map_err(|e| e.into())
    }

    fn get_attributes(&self, context: &Context, id: &GarnetId) -> Option<Vec<u8>> {
        self.inner.provider().get_attributes(context, id)
    }

    fn search_vector(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        if let Some((labels, beta)) = filter {
            let beta_filter = BetaFilter::new(FullPrecision, Arc::new(labels.clone()), beta);
            self.search(&beta_filter, context, query, params, output)
        } else {
            self.search(&FullPrecision, context, query, params, output)
        }
    }

    fn search_element(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        filter: Option<(&GarnetQueryLabelProvider, f32)>,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;
        let mut accessor: provider::FullAccessor<'_, T> =
            <FullPrecision as SearchStrategy<_, _, GarnetId>>::search_accessor(
                &FullPrecision,
                self.inner.provider(),
                context,
            )?;

        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = rt.block_on(accessor.get_element(iid))?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector(context, data_bytes, params, filter, output)
    }

    /// Approach E: Unlimited-effort pure paged post-filter.
    ///
    /// Uses FullPrecision (no BetaFilter) for graph traversal, then applies
    /// the filter callback in a paged loop. The loop does NOT enforce the
    /// max_effort cap — it continues until k results are found or the graph
    /// is exhausted. This allows the search to explore deeply into the graph
    /// at low selectivity, similar to Redis's dual-queue approach.
    fn search_vector_filtered(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        _label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        _max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        let k = params.k_value().get();
        let l_value = params.l_value().get();

        // Always use FullPrecision — no BetaFilter callback overhead during traversal.
        let mut state = self.start_paged_search(FullPrecision, context, query, max(l_value, 256))?;
        paged_filter_loop_unlimited(self, context, &mut state, k, l_value, filter_callback, output, _max_effort)
    }

    fn search_element_filtered(
        &self,
        context: &Context,
        id: &GarnetId,
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;
        let mut accessor: provider::FullAccessor<'_, T> =
            <FullPrecision as SearchStrategy<_, _, GarnetId>>::search_accessor(
                &FullPrecision,
                self.inner.provider(),
                context,
            )?;

        let iid = self.inner.provider().to_internal_id(context, id)?;
        let data = rt.block_on(accessor.get_element(iid))?;
        let data_bytes = bytemuck::cast_slice::<T, u8>(&data);
        self.search_vector_filtered(
            context,
            data_bytes,
            params,
            label_filter,
            filter_callback,
            max_effort,
            output,
        )
    }

    fn remove(&self, context: &Context, id: &GarnetId) -> ANNResult<()> {
        self.inplace_delete(
            FullPrecision,
            context,
            id,
            3,
            InplaceDeleteMethod::TwoHopAndOneHop,
        )
    }

    fn approximate_count(&self) -> u64 {
        self.inner.provider().max_internal_id() as u64
    }

    fn maybe_set_start_point(&self, context: &Context, data: &[u8]) -> ANNResult<()> {
        self.inner
            .provider()
            .maybe_set_start_point(context, bytemuck::cast_slice::<u8, T>(data))
            .map_err(|e| e.into())
    }

    fn internal_id_exists(&self, context: &Context, id: u32) -> bool {
        self.inner.provider().vector_iid_exists(context, id)
    }

    fn external_id_exists(&self, context: &Context, id: &GarnetId) -> bool {
        self.inner.provider().vector_id_exists(context, id)
    }
}

/// Paged search loop WITHOUT max_effort cap.
///
/// Continues fetching pages of candidates from the graph until either:
/// - k matching results are collected, or
/// - the graph frontier is exhausted (next_search_results returns 0)
///
/// This allows the search to explore deeply into the graph for low-selectivity
/// queries, where many candidates need to be evaluated to find matches.
fn paged_filter_loop_unlimited<T, S>(
    index: &DiskANNIndex<GarnetProvider<T>>,
    context: &Context,
    state: &mut SearchState<u32, (S, S::QueryComputer)>,
    k: usize,
    l_value: usize,
    filter_callback: FilterCandidateCallback,
    output: &mut SearchResults<'_>,
    _max_effort: usize,
) -> ANNResult<SearchStats>
where
    T: VectorRepr,
    S: SearchStrategy<GarnetProvider<T>, [T]>,
{
    let mut batch = vec![Neighbor::default(); l_value];
    let mut total_candidates = 0usize;
    let mut total_evaluated = 0usize;
    let mut total_passed = 0usize;
    let mut pages = 0u32;
    let mut graph_ns = 0u64;
    let mut filter_ns = 0u64;
    let mut extid_ns = 0u64;
    let total_start = std::time::Instant::now();
    let mut early_exit = false;

    loop {

        let page_start = std::time::Instant::now();
        let count = index.next_search_results(context, state, l_value, &mut batch)?;
        graph_ns += page_start.elapsed().as_nanos() as u64;
        total_candidates += count;
        pages += 1;

        if count == 0 {
            break;
        }

        for candidate in &batch[..count] {
            total_evaluated += 1;

            let cb_start = std::time::Instant::now();
            let passes = unsafe { filter_callback(context.0, candidate.id) != 0 };
            filter_ns += cb_start.elapsed().as_nanos() as u64;

            if !passes {
                continue;
            }

            total_passed += 1;

            let eid_start = std::time::Instant::now();
            let eid_result = index.inner.provider().to_external_id(context, candidate.id);
            extid_ns += eid_start.elapsed().as_nanos() as u64;

            if let Ok(eid) = eid_result {
                if output.push(eid, candidate.distance).is_full() || output.current_len() >= k {
                    early_exit = true;
                    break;
                }
            }

        }

        if early_exit {
            break;
        }
    }

    let total_us = total_start.elapsed().as_micros();
    let stats = SearchStats { cmps: state.scratch.cmps, hops: state.scratch.hops, result_count: output.current_len() as u32, range_search_second_round: false };
    eprintln!(
        "[paged-unlimited] l={l_value} pages={pages} candidates={total_candidates} evaluated={total_evaluated} passed={total_passed} found={} cmps={} hops={} early_exit={early_exit} total={total_us}µs graph={}µs filter={}µs extid={}µs",
        stats.result_count, stats.cmps, stats.hops, graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
    );
    Ok(stats)
}
