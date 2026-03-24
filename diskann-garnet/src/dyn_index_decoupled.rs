/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach L: Decoupled beam-width / evaluation-budget paged post-filter.
//!
//! Key insight: Redis uses a small beam width (EF=200) for graph traversal
//! and a separate FILTER-EF budget for candidate evaluation. Garnet previously
//! conflated the two: effectiveEF = max(EF, FILTER-EF), making both the beam
//! width AND the priority queue equal to FILTER-EF.
//!
//! This approach decouples them:
//! - Beam width (l_value for paged search) = min(effectiveEF, BEAM_CAP)
//!   where BEAM_CAP=200 matches Redis's default EF.
//! - Evaluation budget = max_effort (= FILTER-EF from C#).
//!
//! With a smaller beam (200 vs 10K), each page is faster (~200 candidates
//! instead of ~10K), but more pages are needed. The graph frontier is
//! narrower but deeper — it follows the nearest-neighbor path more tightly.
//!
//! Combined with the evaluation cap, this should match Redis's latency profile:
//! - At high selectivity: beam=200, ~few hundred candidates → fast
//! - At low selectivity: beam=200, up to FILTER-EF candidates → capped latency

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
use std::sync::Arc;

/// Maximum beam width for the paged search.
/// DiskANN's flat graph structure needs more beam width than HNSW's EF=200
/// because there are no hierarchical skip layers to guide the search.
/// Testing at 100K vectors: beam=200 → ~52% recall, beam=1000 → ~80%, beam=2000 → TBD.
const BEAM_CAP: usize = 2000;

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

    /// Approach L: Decoupled beam-width / evaluation-budget.
    ///
    /// Uses a small beam width (BEAM_CAP=200) for graph traversal, matching
    /// Redis's default EF. The evaluation budget (max_effort = FILTER-EF)
    /// controls how many candidates are checked against the filter. This
    /// produces fast, narrow pages with capped total evaluation cost.
    fn search_vector_filtered(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        _label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);
        let k = params.k_value().get();
        let full_l = params.l_value().get();

        // Decouple: small beam for graph traversal, max_effort for evaluation cap.
        let beam = full_l.min(BEAM_CAP);

        let mut state = self.start_paged_search(FullPrecision, context, query, beam)?;
        paged_filter_loop_capped(self, context, &mut state, k, beam, max_effort, filter_callback, output)
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

/// Paged search loop with evaluation cap.
///
/// Uses a small beam width for fast per-page graph traversal, and caps
/// total candidate evaluation at `max_effort`. Continues fetching pages
/// until either:
/// - k matching results are collected, or
/// - max_effort candidates have been evaluated, or
/// - the graph frontier is exhausted (next_search_results returns 0)
fn paged_filter_loop_capped<T, S>(
    index: &DiskANNIndex<GarnetProvider<T>>,
    context: &Context,
    state: &mut SearchState<u32, (S, S::QueryComputer)>,
    k: usize,
    l_value: usize,
    max_effort: usize,
    filter_callback: FilterCandidateCallback,
    output: &mut SearchResults<'_>,
) -> ANNResult<SearchStats>
where
    T: VectorRepr,
    S: SearchStrategy<GarnetProvider<T>, [T]>,
{
    let mut batch = vec![Neighbor::default(); l_value];
    let mut total_evaluated = 0usize;
    let mut total_passed = 0usize;
    let mut pages = 0u32;
    let mut graph_ns = 0u64;
    let mut filter_ns = 0u64;
    let mut extid_ns = 0u64;
    let total_start = std::time::Instant::now();

    loop {
        let page_start = std::time::Instant::now();
        let count = index.next_search_results(context, state, l_value, &mut batch)?;
        graph_ns += page_start.elapsed().as_nanos() as u64;
        pages += 1;

        if count == 0 {
            break;
        }

        for candidate in &batch[..count] {
            if total_evaluated >= max_effort {
                let total_us = total_start.elapsed().as_micros();
                eprintln!(
                    "[decoupled] beam={l_value} pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
                    output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
                );
                let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
                return Ok(stats);
            }
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
                    let total_us = total_start.elapsed().as_micros();
                    eprintln!(
                        "[decoupled] beam={l_value} pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
                        output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
                    );
                    let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
                    return Ok(stats);
                }
            }
        }
    }

    let total_us = total_start.elapsed().as_micros();
    eprintln!(
        "[decoupled] beam={l_value} pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
        output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
    );
    let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
    Ok(stats)
}
