//! # Çekirdekler Arası İletişim (IPC - Inter-Process Communication)
//! 
//! Bu modül, işletim sisteminin farklı modülleri (`neural-engine`, `logic-gate-core`) 
//! arasındaki iletişimi sağlayan "Merkezi Sinir Sistemi"dir.
//! 
//! Veriler kopyalanmaz (Zero-Copy). Bunun yerine, bir dosya (`.cognitive_bus.mmap`) 
//! doğrudan işletim sisteminin RAM'ine (Memory Map) yansıtılır.
//! Okuma ve yazma işlemleri O(1) donanım hızında gerçekleşir.

use memmap2::MmapMut;
use std::fs::OpenOptions;
use std::path::Path;
use crate::CognitiveSignal;

/// Bellek Otobüsü (Memory Bus) yöneticisi.
pub struct MemoryBus {
    /// İşletim sisteminin RAM'inde açılmış, değiştirilebilir bellek alanı
    mmap: MmapMut,
}

impl MemoryBus {
    /// Yeni bir paylaşımlı bellek dosyası açar veya var olanı RAM'e bağlar.
    /// 
    /// # Argümanlar
    /// * `path` - Belleğe eşlenecek dosyanın sistem yolu (örn: `/tmp/cognitive.bus`)
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(path)?;

        // CognitiveSignal yapısının bellekte kapladığı bayt boyutunu hesapla
        let size = std::mem::size_of::<CognitiveSignal>() as u64;
        
        // Eğer dosya boşsa, işletim sisteminden bu boyutta yer ayırmasını (allocate) iste
        if file.metadata()?.len() < size {
            file.set_len(size)?;
        }

        // DİKKAT: Güvenli olmayan (unsafe) blok. 
        // Dosyayı doğrudan işletim sisteminin sanal belleğine bağlarız.
        let mmap = unsafe { MmapMut::map_mut(&file)? };
        
        Ok(Self { mmap })
    }

    /// Paylaşımlı bellekten güncel komut sinyalini kopyalamadan (zero-copy) okur.
    pub fn read_signal(&self) -> CognitiveSignal {
        let size = std::mem::size_of::<CognitiveSignal>();
        let bytes = &self.mmap[0..size];
        
        // bytemuck ile ham byte dizisini donanım seviyesinde (pod) Struct'a dönüştürür.
        *bytemuck::from_bytes::<CognitiveSignal>(bytes)
    }

    /// Yeni bir sinyali, paylaşımlı bellekteki ilgili alana doğrudan (zero-copy) yazar.
    pub fn write_signal(&mut self, signal: &CognitiveSignal) {
        let bytes = bytemuck::bytes_of(signal);
        
        // Belleğin başından itibaren Struct'ın kapladığı alan kadar veriyi üstüne yaz
        self.mmap[0..bytes.len()].copy_from_slice(bytes);
    }
}