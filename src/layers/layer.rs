use crate::{
    learning_rate::LearningRateManager,
    SharedRef,
    MultiRef,
    conn::connection::Connectable,
    nodes::input::In,
};

pub trait Layer <N, L: LearningRateManager>{
    fn link<T: Connectable + In, O: Layer<T, L>>(&mut self, other: &mut O);
    fn neurons(&self) -> &Vec<MultiRef<N>>;
    fn fresh(neuron_ammount: i32,learning_manager: SharedRef<L>) -> Self;
}