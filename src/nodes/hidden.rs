use super::{
    input::*, 
    output::*
};

struct HiddenNode <N: In> {
    inputs: Vec<f32>,
    threshold: f32,
    outputs: Vec<(f32,InRef<N>)>
}

impl <N: In> In for HiddenNode<N> {
    fn activation(&self, x: f32) -> f32 {
        let result = sigmoid(x);
        if result > self.threshold {
            result
        } else {
            0.0
        }
    }
    fn recieve(&mut self, x: f32) {
        self.inputs.push(x)
    }
}

impl <N: In> Connect<N> for HiddenNode<N> {
    fn connect(&mut self, weight: f32, node: InRef<N>) {
        self.outputs.push((weight, node.clone()))
    }
    
    fn forward_prop(&mut self, x: f32) {
        if x > 0.0 {
            for n in &mut self.outputs {
                n.1.borrow_mut().recieve(x * n.0);
            }
        }
        
    }
}