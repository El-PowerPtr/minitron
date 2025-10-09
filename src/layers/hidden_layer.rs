use crate::{
    nodes::{
        input::In,
        output::Out,
        learn::*,
    },
    learning_rate::LearningRateManager,
    SharedRef,
    MultiRef,
};

use super::layer::*;

struct HiddenLayer<N: In + Out + BackProp, M: LearningRateManager>{
    learning_manager: SharedRef<M>,
    nodes: Vec<MultiRef<N>>,
}