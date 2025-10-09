use super::{
    input::In,
    output::Out,
    learn::Learn,
};
use crate::MultiRef;


pub trait Label<T>: In {
    fn label(&self) -> T;
}

pub struct TrainingLabel<N:Learn + Out, T>{
    label: T,
    expected_val: f32,
    related_node:  MultiRef<N>,
}