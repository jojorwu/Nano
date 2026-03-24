use crate::{Runtime, SimulationSettings};
use std::ffi::{CStr};
use std::os::raw::c_char;

pub struct RuntimeOpaque(Runtime);

#[no_mangle]
pub extern "C" fn genesis_runtime_load(path: *const c_char, backend: *const c_char) -> *mut RuntimeOpaque {
    let path_str = unsafe { CStr::from_ptr(path).to_str().unwrap_or("model.state") };
    let backend_str = unsafe { CStr::from_ptr(backend).to_str().unwrap_or("cpu") };

    let mut settings = SimulationSettings::default();
    settings.preferred_backend = Some(backend_str.to_string());

    match Runtime::load_with_settings(path_str, settings) {
        Ok(rt) => Box::into_raw(Box::new(RuntimeOpaque(rt))),
        Err(_) => std::ptr::null_mut(),
    }
}

#[no_mangle]
pub extern "C" fn genesis_runtime_free(ptr: *mut RuntimeOpaque) {
    if !ptr.is_null() {
        unsafe { let _ = Box::from_raw(ptr); }
    }
}

#[no_mangle]
pub extern "C" fn genesis_runtime_tick(ptr: *mut RuntimeOpaque, inputs: *const i32, output_spikes: *mut bool) {
    let rt = unsafe { &mut (*ptr).0 };
    let n_count = rt.engine.model.neurons.len();
    let input_slice = unsafe { std::slice::from_raw_parts(inputs, n_count) };
    let spikes = rt.tick(input_slice);

    unsafe {
        std::ptr::copy_nonoverlapping(spikes.as_ptr(), output_spikes, n_count);
    }
}

#[no_mangle]
pub extern "C" fn genesis_runtime_inject_text(ptr: *mut RuntimeOpaque, text: *const c_char) {
    let rt = unsafe { &mut (*ptr).0 };
    let text_str = unsafe { CStr::from_ptr(text).to_str().unwrap_or("") };
    rt.inject_text(text_str);
}

#[no_mangle]
pub extern "C" fn genesis_runtime_neuron_count(ptr: *mut RuntimeOpaque) -> u32 {
    let rt = unsafe { &(*ptr).0 };
    rt.engine.model.neurons.len() as u32
}
