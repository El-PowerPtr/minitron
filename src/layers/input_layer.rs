use crate::{
    conn::connection::ConnectsFrom,
    nodes::{
        output::Out,
        input_node::InputNode,
    },
    learning_rate::{
        LearningRateManager,
        Random,
    },
    MultiRef,
    SharedRef,
    new_multi_ref,
};
use super::layer::Layer;

pub struct InputLayer<N: ConnectsFrom, M: LearningRateManager + Random> {
    learning_manager: SharedRef<M>,
    nodes: Vec<MultiRef<InputNode<N>>>,
}

impl <N: ConnectsFrom, M: LearningRateManager + Random> Layer<InputNode<N>, M> for InputLayer<N,M> {
    fn neurons(&self) -> &Vec<MultiRef<InputNode<N>>>{
        &self.nodes
    }
    
    fn fresh(neuron_ammount: i32, learning_manager: SharedRef<M>) -> Self{
        let mut this = InputLayer {
            learning_manager,
            nodes: vec![],
        };
        for _ in 0..neuron_ammount {
            let node = InputNode::new();
            let node_ref = new_multi_ref(node);
            this.nodes.push(node_ref);
        }
        this
    }
    fn pass_info(&mut self) {
        for node in &mut self.nodes {
            (**node).borrow_mut().forward_prop();
        }
    }
}