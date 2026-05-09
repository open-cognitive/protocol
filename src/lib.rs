//! Open-Cognitive OS: Zero-Copy Inter-Process Communication Protocol
//! 
//! Bu kütüphane, `neural-engine` ve `logic-gate-core` arasındaki veri alışverişini
//! serileştirme (serialization) OLMADAN, doğrudan bellek (RAM) üzerinden
//! yapmak için `repr(C)` standartlarında donanım uyumlu veri yapıları tanımlar.

use bytemuck::{Pod, Zeroable};

/// Sistemdeki veri tiplerini belirten sabitler
pub const DTYPE_F32: u8 = 0;
pub const DTYPE_F16: u8 = 1;
pub const DTYPE_BF16: u8 = 2;
pub const DTYPE_INT8: u8 = 3;

/// Bilişsel Çekirdek komut tipleri
pub const CMD_IDLE: u8 = 0;
pub const CMD_FORWARD_PASS: u8 = 1;     // Nöral motordan tahmin isteği
pub const CMD_EVALUATE_LOGITS: u8 = 2;  // Mantık kapısı doğrulama isteği
pub const CMD_HALT: u8 = 255;           // Sistemi durdur

/// Bir Tensor'ün (Çok boyutlu matris) bellekte nerede olduğunu
/// ve boyutlarını tanımlayan harita. 
/// Not: Verinin kendisini taşımaz, sadece bellekteki adresini gösterir (Pointer mantığı).
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct TensorDescriptor {
    pub data_offset: u64,     // Paylaşımlı bellekte (Shared Memory) verinin başladığı bayt
    pub data_size: u64,       // Verinin bayt cinsinden toplam boyutu
    pub dimensions: [u32; 4], // Matris boyutları [Batch, SeqLen, Heads, HeadDim]
    pub dtype: u8,            // Veri tipi (F32, F16 vb.)
    pub _padding: [u8; 7],    // 64-bit bellek hizalaması (Alignment) için boşluk
}

/// Nöral Motor ile Mantık Çekirdeği arasında gidip gelen
/// "Sistem Çağrısı" (System Call) sinyali.
#[repr(C)]
#[derive(Copy, Clone, Debug, Pod, Zeroable)]
pub struct CognitiveSignal {
    pub command_type: u8,       // Hangi komut çalıştırılacak?
    pub cognitive_state: u8,    // O anki ReAct durumu (1=Ingest, 2=Project vb.)
    pub _padding1: [u8; 2],     // Hizalama
    pub context_length: u32,    // Bağlamda (Context) kaç token var?
    pub input_tensor: TensorDescriptor,  // Nöral motora giren veri
    pub output_tensor: TensorDescriptor, // Nöral motorun üreteceği logitlerin yazılacağı adres
}

impl CognitiveSignal {
    /// Yeni bir boş sinyal oluşturur
    pub fn new() -> Self {
        Self {
            command_type: CMD_IDLE,
            cognitive_state: 0,
            _padding1: [0; 2],
            context_length: 0,
            input_tensor: TensorDescriptor::zeroed(),
            output_tensor: TensorDescriptor::zeroed(),
        }
    }
}