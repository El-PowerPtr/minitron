use super::{
    input::In,
    output::Out,
};
use crate::learning_rate::LearningRateManager;
use crate::SharedRef;

pub trait Node<M: LearningRateManager>: In + Out{
    fn new(bias: f32, manager: SharedRef<M>) -> Self;
}