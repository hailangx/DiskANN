/*
 * Copyright (c) Microsoft Corporation.
 * Licensed under the MIT license.
 */

use std::{
    ffi::c_void,
    mem,
    ops::Deref,
    ptr, slice,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};

use diskann::{
    graph::{
        SearchOutputBuffer,
        config::{
            self,
            defaults::{FILTER_BETA, GRAPH_SLACK_FACTOR},
        },
        search,
    },
    utils::VectorRepr,
};
use diskann_providers::index::wrapped_async::DiskANNIndex;
use diskann_quantization::alloc::Poly;
use diskann_vector::distance::Metric;

use crate::{
    alloc::AlignToEight,
    provider::{GarnetProvider, GarnetProviderError},
};
use crate::{
    dyn_index::DynIndex,
    garnet::{
        Callbacks, Context, DeleteCallback, GarnetId, ReadCallback, ReadModifyWriteCallback,
        WriteCallback,
    },
};

mod alloc;
mod dyn_index;
#[cfg(test)]
mod ffi_recall_tests;
#[cfg(test)]
mod ffi_tests;
mod fsm;
mod garnet;
mod labels;
mod provider;
#[cfg(test)]
mod test_utils;

enum IndexState {
    NoStartPoints,
    SettingStartPoints,
    Ready,
}
impl From<usize> for IndexState {
    fn from(value: usize) -> Self {
        assert!(value < 3);
        match value {
            0 => IndexState::NoStartPoints,
            1 => IndexState::SettingStartPoints,
            2 => IndexState::Ready,
            _ => unreachable!(),
        }
    }
}

pub struct Index {
    inner: Box<dyn DynIndex>,
    quant_type: VectorQuantType,
    state: AtomicUsize,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum VectorValueType {
    Invalid = 0,
    FP32,
    XB8,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(C)]
pub enum VectorQuantType {
    Invalid = 0,
    NoQuant,
    Bin,
    Q8,
    XPreQ8,
}

struct SearchResults<'a> {
    ids: &'a mut [u8],
    dists: &'a mut [f32],
    index: usize,
    id_index: usize,
}

/// Per-candidate filter callback: Rust → C#.
/// Returns 1 if candidate passes, 0 otherwise.
/// C# maintains the filter state in thread-local storage, so no state pointer is needed.
pub type FilterCandidateCallback =
    unsafe extern "C" fn(context: u64, internal_id: u32) -> u8;

/// Batch per-candidate filter callback: Rust → C#.
/// Evaluates multiple candidates in one managed/unmanaged transition.
/// Writes 1 or 0 into `results` for each candidate, returns the number that passed.
pub type BatchFilterCandidateCallback =
    Option<unsafe extern "C" fn(context: u64, ids: *const u32, count: u32, results: *mut u8) -> u32>;

/// Per-candidate filter callback with co-located attributes: Rust → C#.
/// Receives attribute bytes directly from co-located storage, eliminating
/// ExtMap and Attribute reads on the C# side.
/// Returns 1 if candidate passes, 0 otherwise.
pub type FilterWithAttrCallback =
    Option<unsafe extern "C" fn(context: u64, internal_id: u32,
                                 attr_ptr: *const u8, attr_len: u32) -> u8>;

/// Batch filter callback with co-located attributes: Rust → C#.
/// Evaluates multiple candidates in one transition, passing attribute data arrays.
/// Writes 1 or 0 into `results` for each candidate, returns the number that passed.
pub type BatchFilterWithAttrCallback =
    Option<unsafe extern "C" fn(context: u64, ids: *const u32,
                                 attr_ptrs: *const *const u8, attr_lens: *const u32,
                                 count: u32, results: *mut u8) -> u32>;

impl SearchResults<'_> {
    fn new(ids: *mut u8, ids_len: usize, dists: *mut f32, dists_len: usize) -> Self {
        let ids = unsafe { slice::from_raw_parts_mut(ids, ids_len) };
        let dists = unsafe { slice::from_raw_parts_mut(dists, dists_len) };
        let index = 0;
        let id_index = 0;
        Self {
            ids,
            dists,
            index,
            id_index,
        }
    }
}

impl SearchOutputBuffer<GarnetId> for SearchResults<'_> {
    fn size_hint(&self) -> Option<usize> {
        Some(self.dists.len() - self.index)
    }

    fn push(&mut self, id: GarnetId, distance: f32) -> diskann::graph::BufferState {
        if self.index >= self.dists.len()
            || self.id_index + mem::size_of::<u32>() + id.len() > self.ids.len()
        {
            return diskann::graph::BufferState::Full;
        }

        let id_len = id.len() as u32;
        self.ids[self.id_index..self.id_index + mem::size_of::<u32>()]
            .copy_from_slice(bytemuck::bytes_of(&id_len));
        self.id_index += mem::size_of::<u32>();

        self.ids[self.id_index..self.id_index + id.len()].copy_from_slice(&id);
        self.dists[self.index] = distance;
        self.index += 1;
        self.id_index += id.len();

        if self.index >= self.dists.len() || self.id_index >= self.ids.len() {
            diskann::graph::BufferState::Full
        } else {
            diskann::graph::BufferState::Available
        }
    }

    fn current_len(&self) -> usize {
        self.index
    }

    fn extend<Itr>(&mut self, itr: Itr) -> usize
    where
        Itr: IntoIterator<Item = (GarnetId, f32)>,
    {
        let initial = self.current_len();

        for (id, dist) in itr {
            if self.push(id, dist).is_full() {
                break;
            }
        }

        self.current_len() - initial
    }
}

fn create_index_impl<T: VectorRepr>(
    quant_type: VectorQuantType,
    config: config::Config,
    dim: usize,
    metric_type: Metric,
    max_degree: usize,
    callbacks: Callbacks,
    context: Context,
) -> Result<Arc<Index>, GarnetProviderError> {
    let provider = GarnetProvider::<T>::new(dim, metric_type, max_degree, callbacks, context)?;
    let state = if provider.start_points_exist() {
        AtomicUsize::new(IndexState::Ready as usize)
    } else {
        AtomicUsize::new(IndexState::NoStartPoints as usize)
    };
    Ok(Arc::new(Index {
        inner: Box::new(DiskANNIndex::new_with_current_thread_runtime(
            config, provider,
        )),
        quant_type,
        state,
    }))
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn create_index(
    ctx: u64,
    dim: u32,
    _reduce_dim: u32,
    quant_type: VectorQuantType,
    metric_type: i32,
    l_build: u32,
    max_degree: u32,
    read_callback: ReadCallback,
    write_callback: WriteCallback,
    delete_callback: DeleteCallback,
    rmw_callback: ReadModifyWriteCallback,
) -> *const c_void {
    let metric_type = match Metric::try_from(metric_type) {
        Ok(m) => m,
        Err(_) => return ptr::null(),
    };

    let target_degree = (max_degree as f32 / GRAPH_SLACK_FACTOR) as usize;
    let config = if let Ok(config) = config::Builder::new(
        target_degree,
        config::MaxDegree::Value(max_degree as usize),
        l_build as usize,
        config::PruneKind::TriangleInequality,
    )
    .build()
    {
        config
    } else {
        return ptr::null();
    };

    let context = Context(ctx);
    let callbacks = Callbacks::new(read_callback, write_callback, delete_callback, rmw_callback);

    match quant_type {
        VectorQuantType::XPreQ8 => {
            if let Ok(index) = create_index_impl::<u8>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
        VectorQuantType::NoQuant => {
            if let Ok(index) = create_index_impl::<f32>(
                quant_type,
                config,
                dim as usize,
                metric_type,
                max_degree as usize,
                callbacks,
                context,
            ) {
                Arc::into_raw(index).cast::<c_void>()
            } else {
                ptr::null()
            }
        }
        _ => ptr::null(),
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn drop_index(_ctx: u64, index_ptr: *const c_void) {
    // SAFETY: Caller must pass in a valid pointer returned from `create_index`.
    let _ = unsafe { Arc::from_raw(index_ptr.cast::<Index>()) };
}

enum PolyCow<'a> {
    Owned(Poly<[u8], AlignToEight>),
    Borrowed(&'a [u8]),
}

impl<'a> Deref for PolyCow<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            PolyCow::Owned(p) => p.deref(),
            PolyCow::Borrowed(p) => p,
        }
    }
}

impl<'a, T: VectorRepr> From<&'a [T]> for PolyCow<'a> {
    fn from(value: &'a [T]) -> Self {
        PolyCow::Borrowed(bytemuck::cast_slice(value))
    }
}

impl<'a> From<Poly<[u8], AlignToEight>> for PolyCow<'a> {
    fn from(value: Poly<[u8], AlignToEight>) -> Self {
        PolyCow::Owned(value)
    }
}

/// Build a combined vector+attributes buffer for co-located storage.
///
/// Layout: `[vector_bytes | attr_len_u32_as_bytes | padded_attr_bytes]`
///
/// - `vector_bytes`: the original vector data (already in the correct format)
/// - `attr_len`: the attribute byte length, stored as a little-endian u32 in sizeof(T) bytes
/// - `padded_attr_bytes`: attribute bytes padded to sizeof(T) alignment
///
/// The element type size depends on quant_type: NoQuant = 4 (f32), XPreQ8 = 1 (u8).
fn build_combined_vector(vector_bytes: &[u8], attr_data: &[u8], quant_type: VectorQuantType) -> Option<Vec<u8>> {
    let elem_size = match quant_type {
        VectorQuantType::NoQuant => mem::size_of::<f32>(),
        VectorQuantType::XPreQ8 => mem::size_of::<u8>(),
        _ => return None,
    };

    let attr_len = attr_data.len();
    // Pad attribute bytes to elem_size boundary
    let padded_attr_len = (attr_len + elem_size - 1) / elem_size * elem_size;

    // Total size: vector + elem_size (for attr_len field) + padded attrs
    let total_len = vector_bytes.len() + elem_size + padded_attr_len;
    let mut buf = vec![0u8; total_len];

    // Copy vector
    buf[..vector_bytes.len()].copy_from_slice(vector_bytes);

    // Write attr_len as u32 in elem_size bytes (little-endian)
    let offset = vector_bytes.len();
    let attr_len_u32 = attr_len as u32;
    let len_bytes = attr_len_u32.to_le_bytes();
    // Copy min(4, elem_size) bytes of the u32
    let copy_len = len_bytes.len().min(elem_size);
    buf[offset..offset + copy_len].copy_from_slice(&len_bytes[..copy_len]);

    // Copy attribute bytes (with zero padding at end)
    let attr_offset = offset + elem_size;
    buf[attr_offset..attr_offset + attr_len].copy_from_slice(attr_data);
    // Remaining padding bytes are already zero from vec![0u8; ...]

    Some(buf)
}

fn interpret_vector<'a>(
    quant_type: VectorQuantType,
    vector_value_type: VectorValueType,
    vector_data: &'a *const u8,
    vector_len: usize,
) -> Option<PolyCow<'a>> {
    let vector_len_bytes = match vector_value_type {
        VectorValueType::FP32 => vector_len * 4,
        VectorValueType::XB8 => vector_len,
        VectorValueType::Invalid => return None,
    };

    let v = unsafe { slice::from_raw_parts(*vector_data, vector_len_bytes) };

    let v = match vector_value_type {
        VectorValueType::Invalid => return None,
        VectorValueType::FP32 => match quant_type {
            VectorQuantType::XPreQ8 => {
                let mut bp = if let Ok(bp) = Poly::broadcast(0u8, vector_len, AlignToEight) {
                    bp
                } else {
                    return None;
                };
                for (idx, e) in bp.iter_mut().enumerate() {
                    let el_size = mem::size_of::<f32>();
                    *e = f32::from_le_bytes(
                        v[idx * el_size..(idx + 1) * el_size].try_into().unwrap(),
                    ) as u8;
                }
                PolyCow::from(bp)
            }
            VectorQuantType::NoQuant if v.as_ptr().align_offset(4) == 0 => {
                // pointer is correctly aligned to interpret as f32
                PolyCow::from(v)
            }
            VectorQuantType::NoQuant => {
                // need to copy f32 data as it is unaligned
                let mut fp = if let Ok(fp) = Poly::broadcast(0u8, vector_len_bytes, AlignToEight) {
                    fp
                } else {
                    return None;
                };
                fp.copy_from_slice(v);
                PolyCow::from(fp)
            }
            _ => {
                return None;
            }
        },
        VectorValueType::XB8 => match quant_type {
            VectorQuantType::XPreQ8 => PolyCow::from(v),
            VectorQuantType::NoQuant => {
                let mut fp = if let Ok(p) =
                    Poly::broadcast(0u8, vector_len_bytes * mem::size_of::<f32>(), AlignToEight)
                {
                    p
                } else {
                    return None;
                };
                for (fe, be) in bytemuck::cast_slice_mut::<u8, f32>(&mut fp)
                    .iter_mut()
                    .zip(v)
                {
                    *fe = *be as f32;
                }
                PolyCow::from(fp)
            }
            _ => return None,
        },
    };

    Some(v)
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn insert(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    vector_value_type: VectorValueType,
    vector_data: *const u8,
    vector_len: usize,
    attribute_data: *const u8,
    attribute_len: usize,
) -> bool {
    if index_ptr.is_null() {
        return false;
    }
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context(ctx);

    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    let v = if let Some(v) = interpret_vector(
        index.quant_type,
        vector_value_type,
        &vector_data,
        vector_len,
    ) {
        v
    } else {
        return false;
    };

    if let Some(_err) =
        ensure_index_ready_or_init(index, || index.inner.maybe_set_start_point(&ctx, &v).err())
    {
        return false;
    };

    // Write attributes to garnet (external store — needed for VGET GETATTR)
    let attr_data = if attribute_len > 0 && !attribute_data.is_null() {
        unsafe { slice::from_raw_parts(attribute_data, attribute_len) }
    } else {
        &[]
    };
    if index.inner.set_attributes(&ctx, &id, attr_data).is_err() {
        return false;
    }

    // Build combined vector + co-located attributes buffer
    // Layout: [vector_bytes | attr_len_as_T | padded_attr_bytes_as_T]
    // This lets filtered search read attrs from the same storage read as the vector.
    let combined = if !attr_data.is_empty() {
        build_combined_vector(&v, attr_data, index.quant_type)
    } else {
        None
    };

    // Insert the pure vector into DiskANN (graph build uses this for distance computation)
    if index.inner.insert(&ctx, &id, &v[..]).is_err() {
        return false;
    }

    // If we have co-located attributes, overwrite the stored vector with the combined buffer.
    // The graph build is done, so the combined data won't affect distance computations.
    // During filtered search, read_multi_lpiid will return the combined data.
    if let Some(ref combined) = combined {
        let internal_id = match index.inner.to_internal_id(&ctx, &id) {
            Some(iid) => iid,
            None => return true, // insert succeeded but can't find internal ID — shouldn't happen
        };
        let combined_as_t: &[u8] = &combined[..];
        if !index.inner.write_vector_raw(&ctx, internal_id, combined_as_t) {
            // Combined write failed, but the insert succeeded with the pure vector.
            // The search will fall back to the old callback path (no co-located attrs).
            eprintln!("[warn] co-located attr write failed for {:?}", id);
        }
    }

    true
}

fn ensure_index_ready_or_init<F, E>(index: &Index, init: F) -> Option<E>
where
    F: FnOnce() -> Option<E>,
{
    // Deal with start point initialization.
    loop {
        match index.state.load(Ordering::Acquire).into() {
            IndexState::Ready => break,                 // Index already ready to go.
            IndexState::SettingStartPoints => continue, // Another thread is setting the start point, wait.
            IndexState::NoStartPoints => {
                // No start points are set yet, so we'll do it.
                match index.state.compare_exchange(
                    IndexState::NoStartPoints as usize,
                    IndexState::SettingStartPoints as usize,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                ) {
                    Ok(_) => {
                        // Run the initializer to set start point.
                        if let Some(e) = init() {
                            // If init() fails, go back to the NoStartPoints state.
                            index
                                .state
                                .store(IndexState::NoStartPoints as usize, Ordering::Release);
                            return Some(e);
                        }
                        index
                            .state
                            .store(IndexState::Ready as usize, Ordering::Release);
                        break;
                    }
                    Err(_) => continue, // Someone else beat us, so wait and try again.
                }
            }
        }
    }
    None
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn set_attribute(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    attribute_data: *const u8,
    attribute_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context(context);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    // Check if the vector exists
    if !index.inner.external_id_exists(&ctx, &id) {
        return false;
    }

    if attribute_len > 0 && !attribute_data.is_null() {
        let attr_data: &[u8] = unsafe { slice::from_raw_parts(attribute_data, attribute_len) };
        if index.inner.set_attributes(&ctx, &id, attr_data).is_err() {
            return false;
        }
    }

    true
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn get_attributes(
    context: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    output_data: *mut u8,
    output_len: usize,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context(context);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    match index.inner.get_attributes(&ctx, &id) {
        Some(data) => {
            if data.len() > output_len {
                return -1;
            }
            let out = unsafe { slice::from_raw_parts_mut(output_data, data.len()) };
            out.copy_from_slice(&data);
            data.len() as i32
        }
        None => -1,
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_vector(
    ctx: u64,
    index_ptr: *const c_void,
    vector_value_type: VectorValueType,
    vector_data: *const u8,
    vector_len: usize,
    _delta: f32,
    search_exploration_factor: u32,
    bitmap_data: *const u8,
    bitmap_len: usize,
    _max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    _continuation: *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    let v = if let Some(v) = interpret_vector(
        index.quant_type,
        vector_value_type,
        &vector_data,
        vector_len,
    ) {
        v
    } else {
        return -1;
    };

    let ctx = Context(ctx);

    let mut output = SearchResults::new(
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let params = match search::Knn::new(
        output_distances_len,
        search_exploration_factor as usize,
        None,
    ) {
        Ok(params) => params,
        Err(_) => return -1,
    };

    let has_filter = !bitmap_data.is_null() && bitmap_len > 0;

    let labels = if has_filter {
        Some(unsafe { labels::GarnetQueryLabelProvider::from_raw(bitmap_data, bitmap_len) })
    } else {
        None
    };
    let filter = labels.as_ref().map(|l| (l, FILTER_BETA));

    let res = index
        .inner
        .search_vector(&ctx, &v, &params, filter, &mut output);

    if let Ok(stats) = res {
        if stats.result_count > i32::MAX as u32 {
            -1
        } else {
            stats.result_count as i32
        }
    } else {
        -1
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_element(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    _delta: f32,
    search_exploration_factor: u32,
    bitmap_data: *const u8,
    bitmap_len: usize,
    _max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    _continuation: *mut c_void,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);
    let ctx = Context(ctx);

    let mut output = SearchResults::new(
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let params = match search::Knn::new(
        output_distances_len,
        search_exploration_factor as usize,
        None,
    ) {
        Ok(params) => params,
        Err(_) => return -1,
    };

    let has_filter = !bitmap_data.is_null() && bitmap_len > 0;
    let labels = if has_filter {
        Some(unsafe { labels::GarnetQueryLabelProvider::from_raw(bitmap_data, bitmap_len) })
    } else {
        None
    };
    let filter = labels.as_ref().map(|l| (l, FILTER_BETA));

    let res = index
        .inner
        .search_element(&ctx, &id, &params, filter, &mut output);
    if let Ok(stats) = res {
        if stats.result_count > i32::MAX as u32 {
            -1
        } else {
            stats.result_count as i32
        }
    } else {
        -1
    }
}

/// # Safety
///
/// FFI — filtered vector search with inline callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_vector_filtered(
    ctx: u64,
    index_ptr: *const c_void,
    vector_value_type: VectorValueType,
    vector_data: *const u8,
    vector_len: usize,
    _delta: f32,
    search_exploration_factor: u32,
    bitmap_data: *const u8,
    bitmap_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    _continuation: *mut c_void,
    filter_callback: FilterCandidateCallback,
    batch_filter_callback: BatchFilterCandidateCallback,
    filter_with_attr_callback: FilterWithAttrCallback,
    batch_filter_with_attr_callback: BatchFilterWithAttrCallback,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    let v = if let Some(v) = interpret_vector(
        index.quant_type,
        vector_value_type,
        &vector_data,
        vector_len,
    ) {
        v
    } else {
        return -1;
    };

    let ctx = Context(ctx);

    let mut output = SearchResults::new(
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let params = match search::Knn::new(
        output_distances_len,
        search_exploration_factor as usize,
        None,
    ) {
        Ok(params) => params,
        Err(_) => return -1,
    };

    let has_filter = !bitmap_data.is_null() && bitmap_len > 0;

    let labels = if has_filter {
        Some(unsafe { labels::GarnetQueryLabelProvider::from_raw(bitmap_data, bitmap_len) })
    } else {
        None
    };
    let filter = labels.as_ref().map(|l| (l, FILTER_BETA));

    let res = index.inner.search_vector_filtered(
        &ctx,
        &v,
        &params,
        filter,
        filter_callback,
        batch_filter_callback,
        filter_with_attr_callback,
        batch_filter_with_attr_callback,
        max_filtering_effort,
        &mut output,
    );

    if let Ok(stats) = res {
        if stats.result_count > i32::MAX as u32 {
            -1
        } else {
            stats.result_count as i32
        }
    } else {
        -1
    }
}

/// # Safety
///
/// FFI — filtered element search with inline callback.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn search_element_filtered(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
    _delta: f32,
    search_exploration_factor: u32,
    bitmap_data: *const u8,
    bitmap_len: usize,
    max_filtering_effort: usize,
    output_ids: *mut u8,
    output_ids_len: usize,
    output_distances: *mut f32,
    output_distances_len: usize,
    _continuation: *mut c_void,
    filter_callback: FilterCandidateCallback,
    batch_filter_callback: BatchFilterCandidateCallback,
    filter_with_attr_callback: FilterWithAttrCallback,
    batch_filter_with_attr_callback: BatchFilterWithAttrCallback,
) -> i32 {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);
    let ctx = Context(ctx);

    let mut output = SearchResults::new(
        output_ids,
        output_ids_len,
        output_distances,
        output_distances_len,
    );

    let params = match search::Knn::new(
        output_distances_len,
        search_exploration_factor as usize,
        None,
    ) {
        Ok(params) => params,
        Err(_) => return -1,
    };

    let has_filter = !bitmap_data.is_null() && bitmap_len > 0;
    let labels = if has_filter {
        Some(unsafe { labels::GarnetQueryLabelProvider::from_raw(bitmap_data, bitmap_len) })
    } else {
        None
    };
    let filter = labels.as_ref().map(|l| (l, FILTER_BETA));

    let res = index.inner.search_element_filtered(
        &ctx,
        &id,
        &params,
        filter,
        filter_callback,
        batch_filter_callback,
        filter_with_attr_callback,
        batch_filter_with_attr_callback,
        max_filtering_effort,
        &mut output,
    );

    if let Ok(stats) = res {
        if stats.result_count > i32::MAX as u32 {
            -1
        } else {
            stats.result_count as i32
        }
    } else {
        -1
    }
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn continue_search(
    _context: u64,
    _index_ptr: *const c_void,
    _continuation: *mut c_void,
    _output_ids: *mut u8,
    _output_ids_len: usize,
    _output_distances: *mut f32,
    _output_distances_len: usize,
    _new_continuation: *mut c_void,
) -> i32 {
    unimplemented!()
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn remove(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context(ctx);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    if !index.inner.external_id_exists(&ctx, &id) {
        return false;
    }

    index.inner.remove(&ctx, &id).is_ok()
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn card(_ctx: u64, index_ptr: *const c_void) -> u64 {
    let index = unsafe { &*index_ptr.cast::<Index>() };

    index.inner.approximate_count()
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_internal_id_valid(
    ctx: u64,
    index_ptr: *const c_void,
    internal_id_data: *const u8,
    internal_id_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context(ctx);
    let internal_id_bytes = unsafe { slice::from_raw_parts(internal_id_data, internal_id_len) };
    if internal_id_bytes.len() != mem::size_of::<u32>() {
        return false;
    }

    let mut id: u32 = 0;
    bytemuck::bytes_of_mut(&mut id).copy_from_slice(internal_id_bytes);

    index.inner.internal_id_exists(&ctx, id)
}

/// # Safety
///
/// FFI
#[unsafe(no_mangle)]
pub unsafe extern "C" fn check_external_id_valid(
    ctx: u64,
    index_ptr: *const c_void,
    id_data: *const u8,
    id_len: usize,
) -> bool {
    let index = unsafe { &*index_ptr.cast::<Index>() };
    let ctx = Context(ctx);
    let id_bytes = unsafe { slice::from_raw_parts(id_data, id_len) };
    let id = GarnetId::from(id_bytes);

    index.inner.external_id_exists(&ctx, &id)
}
