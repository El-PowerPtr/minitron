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
};

pub type MultiRef<N> = Rc<RefCell<Box<N>>>;
pub type SharedRef<N> = Rc<RefCell<N>>;
pub type AMultiRef<N> = Arc<RefCell<Box<N>>>;