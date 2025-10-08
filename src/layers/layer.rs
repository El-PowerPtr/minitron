use crate::{
    learning_rate::LearningRateManager,
};

pub trait Layer <N, L: LearningRateManager>{
    fn link<T>(&mut self, other: &mut impl Layer<T, L>);
}