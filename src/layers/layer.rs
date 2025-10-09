use crate::{
    learning_rate::LearningRateManager,
    SharedRef,
};

pub trait Layer <N, L: LearningRateManager>{
    fn link<T>(&mut self, other: &mut impl Layer<T, L>);
    fn fresh(neuron_ammount: i32,learning_manager: SharedRef<N>) -> Self;
}