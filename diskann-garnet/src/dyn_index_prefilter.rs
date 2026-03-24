/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach D2: Cached-Callback QueryLabelProvider with BetaFilter soft bias.
//!
//! Builds on Attempt 1 (callback-as-QueryLabelProvider) by adding a
//! per-query **result cache** that eliminates redundant FFI calls.
//!
//! During graph traversal, the same internal ID can be encountered as a
//! neighbor of multiple visited nodes, each triggering `is_match()`.
//! The cache stores the result of the first FFI call and returns it
//! instantly on subsequent lookups.
//!
//! The cache is also shared with the paged post-filter loop, so candidates
//! already evaluated during BetaFilter don't need a second FFI call.
//!
//! Cost model:
//!   - First encounter of each ID: ~1.6µs (FFI callback)
//!   - Subsequent encounters: ~30ns (mutex lock + HashMap lookup)
//!   - Total unique FFI calls: ~number of unique neighbors visited

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
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
// Cached FFI Callback as QueryLabelProvider
// ---------------------------------------------------------------------------

/// A `QueryLabelProvider` that calls the FFI `FilterCandidateCallback` and
/// caches results in a `HashMap<u32, bool>`.  Each unique internal ID is
/// evaluated exactly once via FFI; subsequent lookups are O(1) hash lookups.
///
/// The cache is wrapped in a Mutex for Send+Sync compliance. Since DiskANN
/// search is single-threaded, the mutex is always uncontended (~20ns overhead).
#[derive(Debug)]
struct CachedCallbackLabelProvider {
    callback: FilterCandidateCallback,
    context_raw: u64,
    cache: Mutex<HashMap<u32, bool>>,
}

// SAFETY: The callback is a plain function pointer (extern "C" fn), inherently
// Send+Sync.  The context_raw + cache are accessed only from the calling thread.
unsafe impl Send for CachedCallbackLabelProvider {}
unsafe impl Sync for CachedCallbackLabelProvider {}

impl CachedCallbackLabelProvider {
    fn new(callback: FilterCandidateCallback, context_raw: u64) -> Self {
        Self {
            callback,
            context_raw,
            // Pre-allocate for typical search working set
            cache: Mutex::new(HashMap::with_capacity(4096)),
        }
    }

    /// Look up a cached result, or call FFI and cache it.
    #[inline]
    fn is_match_cached(&self, internal_id: u32) -> bool {
        let mut cache = self.cache.lock().unwrap();
        *cache.entry(internal_id).or_insert_with(|| {
            unsafe { (self.callback)(self.context_raw, internal_id) != 0 }
        })
    }

    /// Check the cache without calling FFI.  Returns None if not cached.
    #[inline]
    fn lookup(&self, internal_id: u32) -> Option<bool> {
        let cache = self.cache.lock().unwrap();
        cache.get(&internal_id).copied()
    }
}

impl QueryLabelProvider<u32> for CachedCallbackLabelProvider {
    #[inline(always)]
    fn is_match(&self, internal_id: u32) -> bool {
        self.is_match_cached(internal_id)
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

    /// Approach D2: Cached callback as QueryLabelProvider with BetaFilter.
    ///
    /// Same as Attempt 1 but caches FFI callback results so each internal ID
    /// is evaluated exactly once.  The cache is shared between BetaFilter
    /// (during graph traversal) and the paged post-filter loop.
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
        let l_value = params.l_value().get();

        // Create a caching label provider that shares the cache between
        // BetaFilter (during search) and the paged post-filter loop.
        let label_provider = Arc::new(CachedCallbackLabelProvider::new(
            filter_callback,
            context.0,
        ));

        // BetaFilter uses the cached provider for soft bias during traversal.
        let strategy = BetaFilter::new(FullPrecision, label_provider.clone(), FILTER_BETA);

        // Paged search with BetaFilter soft bias, then cached post-filter.
        let mut state = self.start_paged_search(strategy, context, query, l_value)?;
        paged_filter_loop_cached(
            self,
            context,
            &mut state,
            k,
            l_value,
            max_effort,
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

/// Paged search loop with cached filter: uses the shared CachedCallbackLabelProvider
/// for post-filtering.  Candidates evaluated during BetaFilter's `is_match()` are
/// already cached, so the post-filter check is a free hash lookup.
fn paged_filter_loop_cached<T, S>(
    index: &DiskANNIndex<GarnetProvider<T>>,
    context: &Context,
    state: &mut SearchState<u32, (S, S::QueryComputer)>,
    k: usize,
    l_value: usize,
    max_effort: usize,
    provider: &CachedCallbackLabelProvider,
    output: &mut SearchResults<'_>,
) -> ANNResult<SearchStats>
where
    T: VectorRepr,
    S: SearchStrategy<GarnetProvider<T>, [T]>,
{
    let mut batch = vec![Neighbor::default(); l_value];
    let mut total_evaluated = 0usize;
    let mut pages = 0u32;
    let mut graph_ns = 0u64;
    let mut filter_ns = 0u64;
    let mut cache_hits = 0u64;
    let mut cache_misses = 0u64;
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
                break;
            }
            total_evaluated += 1;

            let cb_start = std::time::Instant::now();
            // Check cache first — BetaFilter likely already evaluated this ID
            let passes = if let Some(cached) = provider.lookup(candidate.id) {
                cache_hits += 1;
                cached
            } else {
                cache_misses += 1;
                provider.is_match_cached(candidate.id)
            };
            filter_ns += cb_start.elapsed().as_nanos() as u64;

            if !passes {
                continue;
            }

            let eid_start = std::time::Instant::now();
            let eid_result = index.inner.provider().to_external_id(context, candidate.id);
            extid_ns += eid_start.elapsed().as_nanos() as u64;

            if let Ok(eid) = eid_result {
                if output.push(eid, candidate.distance).is_full() || output.current_len() >= k {
                    let total_us = total_start.elapsed().as_micros();
                    eprintln!(
                        "[paged-cached-beta] pages={pages} evaluated={total_evaluated} found={} cache_hits={cache_hits} cache_misses={cache_misses} total={total_us}µs graph={}µs filter={}µs extid={}µs",
                        output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
                    );
                    let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
                    return Ok(stats);
                }
            }
        }

        if total_evaluated >= max_effort {
            break;
        }
    }

    let total_us = total_start.elapsed().as_micros();
    eprintln!(
        "[paged-cached-beta] pages={pages} evaluated={total_evaluated} found={} cache_hits={cache_hits} cache_misses={cache_misses} total={total_us}µs graph={}µs filter={}µs extid={}µs",
        output.current_len(), graph_ns / 1000, filter_ns / 1000, extid_ns / 1000
    );
    let stats = SearchStats { cmps: 0, hops: 0, result_count: output.current_len() as u32, range_search_second_round: false };
    Ok(stats)
}
