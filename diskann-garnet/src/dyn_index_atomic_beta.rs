/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach J: Unlimited-effort paged post-filter WITH BetaFilter soft bias
//! and a LOCK-FREE atomic cache for the filter callback.
//!
//! Improvements over Attempt 8 (dyn_index_unlimited_beta.rs):
//! - Replaces Mutex<HashMap<u32, bool>> with Vec<AtomicU8> indexed by internal ID
//! - Each is_match() call is a single atomic load (~1ns) for cache hits
//! - Cache misses do one FFI callback + atomic store
//! - No mutex contention, no hash computation, no allocation during search
//!
//! This should make BetaFilter's is_match() during graph traversal nearly free,
//! allowing the soft bias to steer the search toward matching regions without
//! the overhead that caused Attempt 8 to timeout.

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
use std::sync::atomic::{AtomicU8, Ordering};

/// Beta value for the soft bias — matching candidates get distance × beta.
const FILTER_BETA: f32 = 0.5;

/// Cache states for atomic filter cache.
const UNKNOWN: u8 = 0;
const NO_MATCH: u8 = 1;
const MATCH: u8 = 2;

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
// Lock-free atomic cache for FFI filter callback
// ---------------------------------------------------------------------------

/// A `QueryLabelProvider` that calls the FFI `FilterCandidateCallback` and
/// caches results in a lock-free `Vec<AtomicU8>` indexed by internal ID.
///
/// Each element is one of: UNKNOWN (0), NO_MATCH (1), MATCH (2).
/// Cache lookup is a single atomic load with Relaxed ordering (~1ns).
/// Cache miss triggers one FFI callback + atomic store.
///
/// Memory: ~1 byte per internal ID. For 1M vectors = 1MB.
struct AtomicCacheLabelProvider {
    callback: FilterCandidateCallback,
    context_raw: u64,
    cache: Vec<AtomicU8>,
}

impl std::fmt::Debug for AtomicCacheLabelProvider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AtomicCacheLabelProvider")
            .field("cache_size", &self.cache.len())
            .finish()
    }
}

// SAFETY: callback is a plain function pointer (extern "C" fn), inherently Send+Sync.
// AtomicU8 is Send+Sync. context_raw is only used from the calling thread.
unsafe impl Send for AtomicCacheLabelProvider {}
unsafe impl Sync for AtomicCacheLabelProvider {}

impl AtomicCacheLabelProvider {
    fn new(callback: FilterCandidateCallback, context_raw: u64, max_id: u32) -> Self {
        let size = (max_id as usize).saturating_add(1);
        let mut cache = Vec::with_capacity(size);
        for _ in 0..size {
            cache.push(AtomicU8::new(UNKNOWN));
        }
        Self {
            callback,
            context_raw,
            cache,
        }
    }

    #[inline(always)]
    fn is_match_fast(&self, internal_id: u32) -> bool {
        let idx = internal_id as usize;
        if idx >= self.cache.len() {
            // Out of range — call FFI directly (rare)
            return unsafe { (self.callback)(self.context_raw, internal_id) != 0 };
        }

        let cached = self.cache[idx].load(Ordering::Relaxed);
        if cached != UNKNOWN {
            return cached == MATCH;
        }

        // Cache miss — call FFI and store result
        let result = unsafe { (self.callback)(self.context_raw, internal_id) != 0 };
        self.cache[idx].store(if result { MATCH } else { NO_MATCH }, Ordering::Relaxed);
        result
    }
}

impl QueryLabelProvider<u32> for AtomicCacheLabelProvider {
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

    /// Approach J: Unlimited effort + lock-free atomic BetaFilter.
    ///
    /// Uses BetaFilter with AtomicCacheLabelProvider during graph traversal
    /// to soft-bias toward matching candidates. Then runs the unlimited paged
    /// post-filter loop with the same atomic cache.
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

        // Get max internal ID to size the atomic cache.
        let max_id = self.inner.provider().max_internal_id();

        // Create lock-free atomic cache for filter results.
        let label_provider = Arc::new(AtomicCacheLabelProvider::new(
            filter_callback,
            context.0,
            max_id,
        ));

        // BetaFilter uses the atomic cache for soft bias during traversal.
        let strategy = BetaFilter::new(FullPrecision, label_provider.clone(), FILTER_BETA);

        // Paged search with BetaFilter soft bias, then UNLIMITED post-filter.
        let mut state = self.start_paged_search(strategy, context, query, l_value)?;
        paged_filter_loop_unlimited_atomic(
            self,
            context,
            &mut state,
            k,
            l_value,
            &label_provider,
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

/// Paged search loop WITHOUT max_effort cap, using atomic cached filter.
///
/// Continues fetching pages of candidates from the graph until either:
/// - k matching results are collected, or
/// - the graph frontier is exhausted (next_search_results returns 0)
///
/// Uses AtomicCacheLabelProvider so candidates already evaluated by BetaFilter
/// during graph traversal get O(1) atomic load instead of FFI calls.
fn paged_filter_loop_unlimited_atomic<T, S>(
    index: &DiskANNIndex<GarnetProvider<T>>,
    context: &Context,
    state: &mut SearchState<u32, (S, S::QueryComputer)>,
    k: usize,
    l_value: usize,
    provider: &AtomicCacheLabelProvider,
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
            let passes = provider.is_match_fast(candidate.id);
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
                        "[atomic-beta] pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
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
        "[atomic-beta] pages={pages} evaluated={total_evaluated} passed={total_passed} found={} total={total_us}µs graph={}µs filter={}µs extid={}µs",
        output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
    );
    let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
    Ok(stats)
}
