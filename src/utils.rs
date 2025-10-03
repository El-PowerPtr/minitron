use std::{
    boxed::Box,
    rc::Rc,
    cell::RefCell,
    std::sync::Arc,
};

pub type MultiRef<N> = Rc<RefCell<Box<N>>>;

pub type AMultiRef<N> = Arc<RefCell<Box<N>>>;