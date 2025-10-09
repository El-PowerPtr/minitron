use crate::{
    learning_rate::LearningRateManager,
    SharedRef,
};

pub trait Layer <N, L: LearningRateManager>{
    fn link<T>(&mut self, other: &mut impl Layer<T, L>);
    fn fresh(learning_manager: SharedRef<N>) -> Self;
}