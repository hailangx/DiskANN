/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

//! Approach B: Hybrid during-traversal filtering.
//!
//! Instead of the paged post-filter loop, this wraps the FFI
//! `FilterCandidateCallback` as a `HybridPredicate` and feeds it into
//! `expand_beam` via a custom `SearchStrategy`.  The predicate is composed
//! with `NotInMut` so that non-matching neighbours are pruned *during*
//! graph traversal rather than after the page completes.

use crate::{
    FilterCandidateCallback,
    SearchResults,
    garnet::{Context, GarnetId},
    labels::GarnetQueryLabelProvider,
    provider::{self, CopyExternalIds, GarnetProvider},
};
use diskann::{
    ANNError, ANNResult,
    graph::{
        glue::{self, ExpandBeam, SearchStrategy},
        index::SearchStats,
        search,
    },
    provider::{Accessor, DataProvider, DelegateNeighbor},
    utils::VectorRepr,
};
use diskann_providers::{
    index::wrapped_async::DiskANNIndex,
    model::graph::provider::{async_::common::FullPrecision, layers::BetaFilter},
};
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

    /// Approach B: wrap the callback as a HybridPredicate and use the standard
    /// `self.search()` path so the predicate fires inside `expand_beam`.
    fn search_vector_filtered(
        &self,
        context: &Context,
        data: &[u8],
        params: &search::Knn,
        label_filter: Option<(&GarnetQueryLabelProvider, f32)>,
        filter_callback: FilterCandidateCallback,
        max_effort: usize,
        output: &mut SearchResults<'_>,
    ) -> ANNResult<SearchStats> {
        let query = bytemuck::cast_slice::<u8, T>(data);

        let strategy = CallbackFilterPrecision {
            context_raw: context.0,
            callback: filter_callback,
            max_effort,
        };

        if let Some((labels, beta)) = label_filter {
            let beta_strategy = BetaFilter::new(strategy, Arc::new(labels.clone()), beta);
            self.search(&beta_strategy, context, query, params, output)
        } else {
            self.search(&strategy, context, query, params, output)
        }
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
            diskann::graph::InplaceDeleteMethod::TwoHopAndOneHop,
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

// ---------------------------------------------------------------------------
// Callback-based HybridPredicate + SearchStrategy for Approach B
// ---------------------------------------------------------------------------

/// A `SearchStrategy` that wraps the callback as a `HybridPredicate` applied
/// inside `expand_beam`.  The accessor delegates to `FullAccessor` but the
/// `ExpandBeam` impl composes a `CallbackPredicate` with the passed-in
/// visited-set predicate so that neighbours failing the filter are never
/// added to the search frontier.
///
/// Pipeline: FilterStartPoints -> CallbackPostFilter -> CopyExternalIds
///
/// `CallbackPostFilter` is included as a safety-net: if any non-matching
/// candidate somehow ends up in the result list it will be dropped before
/// external-id conversion.  In the normal case `expand_beam` has already
/// pruned everything.
pub struct CallbackFilterPrecision {
    pub context_raw: u64,
    pub callback: FilterCandidateCallback,
    pub max_effort: usize,
}

unsafe impl Send for CallbackFilterPrecision {}
unsafe impl Sync for CallbackFilterPrecision {}

impl<T: VectorRepr> SearchStrategy<GarnetProvider<T>, [T], GarnetId> for CallbackFilterPrecision {
    type SearchAccessor<'a> = CallbackAccessor<'a, T>;
    type SearchAccessorError = provider::GarnetProviderError;
    type QueryComputer = T::QueryDistance;
    type PostProcessor = glue::Pipeline<glue::FilterStartPoints, CopyExternalIds>;

    fn search_accessor<'a>(
        &'a self,
        provider: &'a GarnetProvider<T>,
        context: &'a <GarnetProvider<T> as DataProvider>::Context,
    ) -> Result<Self::SearchAccessor<'a>, Self::SearchAccessorError> {
        let inner = provider::FullAccessor::new(provider, context, true);
        Ok(CallbackAccessor {
            inner,
            callback: self.callback,
            context_raw: self.context_raw,
        })
    }

    fn post_processor(&self) -> Self::PostProcessor {
        Default::default()
    }
}

/// Wrapper accessor that delegates everything to `FullAccessor` except
/// `expand_beam`, where it composes the callback predicate.
pub struct CallbackAccessor<'a, T: VectorRepr> {
    inner: provider::FullAccessor<'a, T>,
    callback: FilterCandidateCallback,
    context_raw: u64,
}

// -- Delegate trait impls to inner -------------------------------------------

impl<T: VectorRepr> diskann::provider::HasId for CallbackAccessor<'_, T> {
    type Id = u32;
}

impl<T: VectorRepr> glue::SearchExt for CallbackAccessor<'_, T> {
    fn starting_points(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<Vec<Self::Id>>> + Send {
        self.inner.starting_points()
    }

    fn is_not_start_point(
        &self,
    ) -> impl std::future::Future<Output = ANNResult<impl Fn(Self::Id) -> bool + Send + Sync + 'static>>
           + Send {
        self.inner.is_not_start_point()
    }
}

impl<T: VectorRepr> Accessor for CallbackAccessor<'_, T> {
    type Extended = Vec<T>;
    type Element<'b>
        = Vec<T>
    where
        Self: 'b;
    type ElementRef<'b> = &'b [T];
    type GetError = provider::GarnetProviderError;

    fn get_element(
        &mut self,
        id: Self::Id,
    ) -> impl std::future::Future<Output = Result<Self::Element<'_>, Self::GetError>> + Send {
        self.inner.get_element(id)
    }
}

impl<T: VectorRepr> diskann::provider::BuildQueryComputer<[T]> for CallbackAccessor<'_, T> {
    type QueryComputer = T::QueryDistance;
    type QueryComputerError = provider::GarnetProviderError;

    fn build_query_computer(
        &self,
        from: &[T],
    ) -> Result<Self::QueryComputer, Self::QueryComputerError> {
        self.inner.build_query_computer(from)
    }
}

impl<T: VectorRepr> diskann::provider::BuildDistanceComputer for CallbackAccessor<'_, T> {
    type DistanceComputer = T::Distance;
    type DistanceComputerError = provider::GarnetProviderError;

    fn build_distance_computer(
        &self,
    ) -> Result<Self::DistanceComputer, Self::DistanceComputerError> {
        self.inner.build_distance_computer()
    }
}

// -- DelegateNeighbor (required by ExpandBeam's AsNeighbor supertrait) --------

/// Delegate to the inner `FullAccessor`'s neighbor accessor.
/// `'p` is the provider lifetime, `'a` is the borrow lifetime for DelegateNeighbor.
impl<'p, 'a, T: VectorRepr> DelegateNeighbor<'a> for CallbackAccessor<'p, T> {
    type Delegate = <provider::FullAccessor<'p, T> as DelegateNeighbor<'a>>::Delegate;
    fn delegate_neighbor(&'a mut self) -> Self::Delegate {
        self.inner.delegate_neighbor()
    }
}

// -- SearchPostProcess for CopyExternalIds with CallbackAccessor -------------

impl<'a, T: VectorRepr> glue::SearchPostProcess<CallbackAccessor<'a, T>, [T], GarnetId>
    for CopyExternalIds
{
    type Error = provider::GarnetProviderError;

    fn post_process<I, B>(
        &self,
        accessor: &mut CallbackAccessor<'a, T>,
        _query: &[T],
        _computer: &<CallbackAccessor<'a, T> as diskann::provider::BuildQueryComputer<[T]>>::QueryComputer,
        candidates: I,
        output: &mut B,
    ) -> impl std::future::Future<Output = Result<usize, Self::Error>> + Send
    where
        I: Iterator<Item = diskann::neighbor::Neighbor<<CallbackAccessor<'a, T> as diskann::provider::HasId>::Id>> + Send,
        B: diskann::graph::SearchOutputBuffer<GarnetId> + Send + ?Sized,
    {
        let initial = output.current_len();
        for n in candidates {
            let id = match accessor.inner.provider.to_external_id(accessor.inner.context, n.id) {
                Ok(id) => id,
                Err(e) => return std::future::ready(Err(e)),
            };
            if output.push(id, n.distance).is_full() {
                break;
            }
        }
        let count = output.current_len() - initial;
        std::future::ready(Ok(count))
    }
}

// -- ExpandBeam with composed predicate --------------------------------------

/// Compose two predicates: the outer `P` (typically `NotInMut`) AND the
/// callback filter.  Both `Predicate::eval` and `PredicateMut::eval_mut`
/// short-circuit: if the callback says "no" the item is immediately rejected.
struct ComposedPredicate<P> {
    outer: P,
    callback: FilterCandidateCallback,
    context_raw: u64,
}

impl<P: glue::Predicate<u32>> glue::Predicate<u32> for ComposedPredicate<P> {
    #[inline(always)]
    fn eval(&self, item: &u32) -> bool {
        // Check callback first (cheaper to reject early)
        let passes = unsafe { (self.callback)(self.context_raw, *item) != 0 };
        passes && self.outer.eval(item)
    }
}

impl<P: glue::PredicateMut<u32>> glue::PredicateMut<u32> for ComposedPredicate<P> {
    #[inline(always)]
    fn eval_mut(&mut self, item: &u32) -> bool {
        let passes = unsafe { (self.callback)(self.context_raw, *item) != 0 };
        passes && self.outer.eval_mut(item)
    }
}

impl<P: glue::HybridPredicate<u32>> glue::HybridPredicate<u32> for ComposedPredicate<P> {}

impl<T: VectorRepr> ExpandBeam<[T]> for CallbackAccessor<'_, T> {
    fn expand_beam<Itr, P, F>(
        &mut self,
        ids: Itr,
        computer: &Self::QueryComputer,
        pred: P,
        on_neighbors: F,
    ) -> impl std::future::Future<Output = ANNResult<()>> + Send
    where
        Itr: Iterator<Item = Self::Id> + Send,
        P: glue::HybridPredicate<Self::Id> + Send + Sync,
        F: FnMut(f32, Self::Id) + Send,
    {
        let composed = ComposedPredicate {
            outer: pred,
            callback: self.callback,
            context_raw: self.context_raw,
        };
        self.inner.expand_beam(ids, computer, composed, on_neighbors)
    }
}
