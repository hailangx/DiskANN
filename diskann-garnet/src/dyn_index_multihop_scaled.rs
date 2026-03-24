/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach H: MultihopSearch with Scaled L_value.
//!
//! Same as Approach G (MultihopSearch + CachedCallbackLabelProvider) but
//! with a critical fix: uses `max_effort` (FILTER-EF) as the L_value
//! for the search instead of the base EF parameter (200).
//!
//! In Attempt 9, MultihopSearch terminated after exploring only L=200
//! candidates, achieving 1% recall at low selectivity. By setting
//! L_value = FILTER-EF (1K-50K), the search explores many more candidates
//! before terminating, giving the two-hop exploration enough room to find
//! matching nodes in sparse filter spaces.
//!
//! The `CachedCallbackLabelProvider` wraps the FFI filter callback with
//! a HashMap cache, so each internal ID is evaluated exactly once via FFI.
//! During two-hop expansion, the predicate calls `is_match()` which hits
//! the cache for previously-seen IDs (O(1) hash lookup instead of FFI).

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
        index::{QueryLabelProvider, SearchStats},
        search::{self, MultihopSearch},
    },
    provider::{Accessor, DataProvider},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

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
/// caches results in a `HashMap<u32, bool>`.
#[derive(Debug)]
struct CachedCallbackLabelProvider {
    callback: FilterCandidateCallback,
    context_raw: u64,
    cache: Mutex<HashMap<u32, bool>>,
}

unsafe impl Send for CachedCallbackLabelProvider {}
unsafe impl Sync for CachedCallbackLabelProvider {}

impl CachedCallbackLabelProvider {
    fn new(callback: FilterCandidateCallback, context_raw: u64) -> Self {
        Self {
            callback,
            context_raw,
            cache: Mutex::new(HashMap::with_capacity(4096)),
        }
    }

    #[inline]
    fn is_match_cached(&self, internal_id: u32) -> bool {
        let mut cache = self.cache.lock().unwrap();
        *cache.entry(internal_id).or_insert_with(|| {
            unsafe { (self.callback)(self.context_raw, internal_id) != 0 }
        })
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

    /// Approach H: MultihopSearch with scaled L_value.
    ///
    /// Uses DiskANN's native multihop search with L_value set to max_effort
    /// (FILTER-EF) instead of the base EF. This gives the search enough
    /// exploration budget to find matching nodes at low selectivity.
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

        // Create a caching label provider for the FFI callback.
        let label_provider = CachedCallbackLabelProvider::new(
            filter_callback,
            context.0,
        );

        // Use max_effort (FILTER-EF) as L_value instead of the base EF.
        // This is the key difference from Attempt 9: the search explores
        // max_effort candidates instead of just 200.
        let l_value = if max_effort > 0 {
            max_effort
        } else {
            params.l_value().get()
        };
        let scaled_params = search::Knn::new(
            params.k_value().get(),
            l_value,
            Some(params.beam_width().get()),
        ).map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;

        // Construct MultihopSearch with scaled params and the label provider.
        let multihop = MultihopSearch::new(scaled_params, &label_provider);

        // Use the inner async index directly since the wrapped sync `search()`
        // only accepts Knn, not MultihopSearch.
        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .map_err(|e| ANNError::new(diskann::ANNErrorKind::Opaque, e))?;

        let stats = rt.block_on(
            self.inner.search(multihop, &FullPrecision, context, query, output)
        )?;

        Ok(stats)
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
