//! Open-Cognitive OS: Zero-Copy Inter-Process Communication Protocol

pub mod ipc;
use bytemuck::{Pod, Zeroable};

pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
pub const DTYPE_INT8: u8 = 3;

// --- ÇEKİRDEK KOMUTLARI ---
pub const CMD_IDLE: u8 = 0;
pub const CMD_FORWARD_PASS: u8 = 1;     
pub const CMD_EVALUATE_LOGITS: u8 = 2;  
pub const CMD_EXECUTE_TOOL: u8 = 3;     
pub const CMD_HALT: u8 = 255;           

// --- WASM TOOL ID'LERİ (Sistemdeki araçların donanım seviyesi karşılıkları) ---
pub const TOOL_SQUARE: u32 = 1; // Parametrenin karesini alma aracı

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TensorDescriptor {
    pub data_offset: u64,     
    pub data_size: u64,       
    pub dimensions: [u32; 4], 
    pub dtype: u8,            
    pub _padding: [u8; 7],    
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CognitiveSignal {
    pub command_type: u8,       
    pub cognitive_state: u8,    
    pub _padding1: [u8; 2],     
    pub context_length: u32,    
    pub input_tensor: TensorDescriptor,  
    pub output_tensor: TensorDescriptor, 
    pub prompt_buffer: [u8; 512], // String/Metin girdisi için C-String tamponu
    pub payload: [u8; 256],       // Zero-Copy araç parametreleri ve sonuçları için alan
}

impl CognitiveSignal {
    pub fn new() -> Self {
        Self {
            command_type: CMD_IDLE,
            cognitive_state: 0,
            _padding1: [0; 2],
            context_length: 0,
            input_tensor: TensorDescriptor::zeroed(),
            output_tensor: TensorDescriptor::zeroed(),
            prompt_buffer: [0; 512],
            payload: [0; 256],
        }
    }

    /// String'i güvenli bir şekilde belleğe yazar
    pub fn set_prompt(&mut self, text: &str) {
        let bytes = text.as_bytes();
        let len = bytes.len().min(512);
        self.prompt_buffer[..len].copy_from_slice(&bytes[..len]);
        if len < 512 {
            self.prompt_buffer[len] = 0; // C-style null terminator
        }
    }

    /// Bellekteki metni okur
    pub fn get_prompt(&self) -> String {
        let mut end = 0;
        while end < 512 && self.prompt_buffer[end] != 0 {
            end += 1;
        }
        String::from_utf8_lossy(&self.prompt_buffer[..end]).into_owned()
    }

    /// Aracın ID'sini (Tool ID) ve girdisini payload'a yazar (i64 destekli)
    pub fn set_tool_call(&mut self, tool_id: u32, input: i64) {
        let id_bytes = tool_id.to_le_bytes();
        let input_bytes = input.to_le_bytes();
        self.payload[0..4].copy_from_slice(&id_bytes);
        self.payload[4..12].copy_from_slice(&input_bytes); // 8 byte
    }

    /// Payload'dan Tool ID ve girdisini donanım hızında okur.
    pub fn get_tool_call(&self) -> (u32, i64) {
        let tool_id = u32::from_le_bytes(self.payload[0..4].try_into().unwrap());
        let input = i64::from_le_bytes(self.payload[4..12].try_into().unwrap());
        (tool_id, input)
    }

    /// Aracın WebAssembly Sandbox'ından dönen sonucunu payload'a kaydeder.
    pub fn set_tool_result(&mut self, result: i64) {
        let res_bytes = result.to_le_bytes();
        self.payload[12..20].copy_from_slice(&res_bytes); // Sonraki 8 byte
    }

    /// Aracın ürettiği sonucu okur.
    pub fn get_tool_result(&self) -> i64 {
        i64::from_le_bytes(self.payload[12..20].try_into().unwrap())
    }
}