/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach K: Pre-computed bitmap BetaFilter + unlimited paged post-filter.
//!
//! Key insight: Attempts 8 and 12 used BetaFilter with garbage label data
//! (raw filter expression bytes interpreted as a bitset). This caused random
//! distance biasing, making graph traversal SLOWER.
//!
//! This approach fixes that by PRE-COMPUTING a correct bitmap:
//! 1. Before search, call filter_callback for ALL live internal IDs
//! 2. Build a Vec<bool> indexed by internal ID
//! 3. Wrap as QueryLabelProvider for BetaFilter — correct soft bias!
//! 4. Use BetaFilter during graph traversal to steer toward matching regions
//! 5. Use same bitmap for post-filter check (O(1) lookup, no FFI callback)
//!
//! Expected: BetaFilter with correct data should reduce candidates explored,
//! while bitmap post-filter eliminates per-candidate FFI callback cost.

use crate::{
    FilterCandidateCallback,
    SearchResults,
    garnet::{Context, GarnetId},
    labels::GarnetQueryLabelProvider,
    provider::{self, GarnetProvider},
};
use diskann::{
    ANNError, ANNResult,
    graph::{
        InplaceDeleteMethod, SearchOutputBuffer,
        glue::SearchStrategy,
        index::{QueryLabelProvider, SearchState, SearchStats},
        search,
    },
    neighbor::Neighbor,
    provider::{Accessor, DataProvider},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};
use std::sync::Arc;

/// Beta value for the soft bias — matching candidates get distance × beta.
const FILTER_BETA: f32 = 0.5;

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

// ---------------------------------------------------------------------------
// Pre-computed bitmap QueryLabelProvider
// ---------------------------------------------------------------------------

/// A `QueryLabelProvider` backed by a pre-computed `Vec<bool>` bitmap.
///
/// Before search begins, we call `filter_callback` for every live internal ID
/// and store the results. During graph traversal, BetaFilter calls `is_match()`
/// which is a simple `Vec<bool>` index lookup (~1ns).
///
/// Memory: 1 byte per internal ID. For 1M vectors = 1MB.
struct BitmapLabelProvider {
    bitmap: Vec<bool>,
    match_count: u32,
}

impl std::fmt::Debug for BitmapLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BitmapLabelProvider")
            .field("size", &self.bitmap.len())
            .field("matches", &self.match_count)
            .finish()
    }
}

// SAFETY: Vec<bool> is inherently Send+Sync (read-only after construction).
unsafe impl Send for BitmapLabelProvider {}
unsafe impl Sync for BitmapLabelProvider {}

impl BitmapLabelProvider {
    /// Build the bitmap by evaluating the filter callback for all internal IDs.
    ///
    /// Calls filter_callback for each internal ID from 0..max_id.
    /// The callback itself handles non-existent IDs (returns 0/false).
    fn build(
        callback: FilterCandidateCallback,
        context_raw: u64,
        max_id: u32,
    ) -> Self {
        let size = (max_id as usize).saturating_add(1);
        let mut bitmap = vec![false; size];
        let mut match_count = 0u32;

        for iid in 0..max_id {
            let passes = unsafe { (callback)(context_raw, iid) != 0 };
            if passes {
                bitmap[iid as usize] = true;
                match_count += 1;
            }
        }

        Self { bitmap, match_count }
    }

    #[inline(always)]
    fn is_match_fast(&self, internal_id: u32) -> bool {
        let idx = internal_id as usize;
        if idx < self.bitmap.len() {
            self.bitmap[idx]
        } else {
            false
        }
    }
}

impl QueryLabelProvider<u32> for BitmapLabelProvider {
    #[inline(always)]
    fn is_match(&self, internal_id: u32) -> bool {
        self.is_match_fast(internal_id)
    }
}

// ---------------------------------------------------------------------------
// DynIndex implementation
// ---------------------------------------------------------------------------

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

    /// Approach K: Pre-computed bitmap BetaFilter + unlimited post-filter.
    ///
    /// 1. Pre-compute bitmap by calling filter_callback for all live internal IDs
    /// 2. Use bitmap as QueryLabelProvider for BetaFilter during graph traversal
    /// 3. Run unlimited paged post-filter with bitmap lookup (no FFI callbacks)
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

        // Step 1: Pre-compute bitmap by evaluating filter on all internal IDs.
        let max_id = self.inner.provider().max_internal_id();
        let bitmap_start = std::time::Instant::now();
        let bitmap = Arc::new(BitmapLabelProvider::build(
            filter_callback,
            context.0,
            max_id,
        ));
        let bitmap_us = bitmap_start.elapsed().as_micros();
        eprintln!(
            "[prefilter-bitmap] built bitmap: max_id={max_id} matches={} time={bitmap_us}µs",
            bitmap.match_count
        );

        // Step 2: BetaFilter with CORRECT label data for soft bias during traversal.
        let strategy = BetaFilter::new(FullPrecision, bitmap.clone(), FILTER_BETA);

        // Step 3: Paged search with BetaFilter, then unlimited bitmap post-filter.
        let mut state = self.start_paged_search(strategy, context, query, l_value)?;
        paged_filter_loop_bitmap(
            self,
            context,
            &mut state,
            k,
            l_value,
            &bitmap,
            output,
        )
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

/// Paged search loop WITHOUT max_effort cap, using pre-computed bitmap.
///
/// Continues fetching pages of candidates from the graph until either:
/// - k matching results are collected, or
/// - the graph frontier is exhausted (next_search_results returns 0)
///
/// Uses BitmapLabelProvider for O(1) filter lookup — no FFI callbacks needed.
fn paged_filter_loop_bitmap<T, S>(
    index: &DiskANNIndex<GarnetProvider<T>>,
    context: &Context,
    state: &mut SearchState<u32, (S, S::QueryComputer)>,
    k: usize,
    l_value: usize,
    bitmap: &BitmapLabelProvider,
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
            total_evaluated += 1;

            let cb_start = std::time::Instant::now();
            let passes = bitmap.is_match_fast(candidate.id);
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
                        "[prefilter-bitmap] pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
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
        "[prefilter-bitmap] pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
        output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
    );
    let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
    Ok(stats)
}
