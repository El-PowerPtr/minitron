#![allow(dead_code)]
mod nodes;
mod layers;
mod conn;
mod learning_rate;

use std::{
    boxed::Box,
    rc::Rc,
    cell::RefCell,
    sync::Arc,
    cmp::Ordering,
};

pub type MultiRef<N> = Rc<RefCell<Box<N>>>;
pub type SharedRef<N> = Rc<RefCell<N>>;
pub type AMultiRef<N> = Arc<RefCell<Box<N>>>;

pub fn new_multi_ref<N>(val: N) -> MultiRef<N>{
    Rc::new(RefCell::new(Box::new(val)))
}

pub fn new_shared_ref<N>(val: N) -> SharedRef<N>{
    Rc::new(RefCell::new(val))
}

#[must_use]
pub fn compare_floats(a: &f32, b: &f32) -> Ordering {
    let diff = a - b;
    if diff.abs() <= 0.000_000_000_001 {
        Ordering::Equal
    } else if diff > 0.0 {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}