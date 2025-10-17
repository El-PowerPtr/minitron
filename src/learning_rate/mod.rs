use std::time::{SystemTime, Duration, UNIX_EPOCH};

pub trait LearningRateManager {
    fn learning_rate(&self) -> f32;
}

pub trait Random {
    fn rand_float(&mut self) -> f32;
}

pub struct RandomGen{
    seed: u32,
}

impl RandomGen{
    #[inline]
    fn next_step(&mut self) -> u32{
        self.seed ^= self.seed << 7;
        self.seed ^= self.seed >> 13;
        self.seed ^= self.seed << 21;
        self.seed ^= 0xF10A32C5;
        
        self.seed
    }
    
    pub fn rand_float(&mut self) -> f32{
        (self.next_step() as f32) / 1.0001
    }
    
    pub fn new() -> RandomGen{
        RandomGen {
            seed: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO)
                    .as_nanos() as u32
        }
    }
}