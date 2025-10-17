use crate::{
    new_multi_ref,
    new_shared_ref,
    nodes::{
        input::In,
        node::Node,
        learn::*,
    },
    conn::connection::*,
    learning_rate::{
        LearningRateManager,
        Random,
    },
    SharedRef,
    MultiRef,
};

use super::layer::*;

struct HiddenLayer<N: Connectable + Node<M> + Learn + BackProp, M: LearningRateManager + Random>{
    learning_manager: SharedRef<M>,
    nodes: Vec<MultiRef<N>>,
}

impl <N: Node<M> + Connectable + Learn + BackProp, M: LearningRateManager + Random> HiddenLayer<N,M>{
    fn learn(&mut self){
        for i in &mut self.nodes {
            (**i).borrow_mut().learn();
            (**i).borrow_mut().send_feedback();
        }
    }
}

impl <N: Connectable + Node<M> + Learn + BackProp, M: LearningRateManager + Random> Layer<N,M> for HiddenLayer<N,M> {
    fn link<T: Connectable + In, O: Layer<T,M>>(&mut self, other: &mut O){
        for other_node in other.neurons() {
            for self_node in &mut self.nodes {
                let conn = Connection::new(self_node.clone(), other_node.clone(),self.learning_manager.borrow_mut().rand_float());
                let conn_ref = new_shared_ref(conn);
                (**self_node).borrow_mut().add_connection(conn_ref.clone());
                (**self_node).borrow_mut().add_connection(conn_ref.clone());
                (**other_node).borrow_mut().add_connection(conn_ref.clone());
            }
        }
    }
    
    fn neurons(&self) -> &Vec<MultiRef<N>>{
        &self.nodes
    }
    
    fn fresh(neuron_ammount: i32,learning_manager: SharedRef<M>) -> Self{
        let mut this = HiddenLayer {
            learning_manager,
            nodes: vec![],
        };
        for _ in 0..neuron_ammount {
            let node = N::new(this.learning_manager.borrow_mut().rand_float(), this.learning_manager.clone());
            let node_ref = new_multi_ref(node);
            this.nodes.push(node_ref);
        }
        this
    }
}