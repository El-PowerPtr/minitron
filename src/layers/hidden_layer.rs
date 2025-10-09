use crate::{
    nodes::{
        input::In,
        output::Out,
        learn::*,
    },
    conn::connect::Connect,
    learning_rate::LearningRateManager,
    SharedRef,
    MultiRef,
};

use super::layer::*;

struct HiddenLayer<N: In + Out + Connect + BackProp, M: LearningRateManager>{
    learning_manager: SharedRef<M>,
    nodes: Vec<MultiRef<N>>,
}