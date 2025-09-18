pub (mod) trait Out{
    fn activation(&self) -> f32;
}

pub (mod) trait Connect<N: In>{
    fn connect(&mut self, weight: f32, node: &N) -> bool;
}

fn sigmoid(x: &f32) -> f32 {
    1 / (1 + exp(x))
}