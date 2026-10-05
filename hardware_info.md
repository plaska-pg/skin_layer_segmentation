# Hardware Information

Collected from Windows on 2026-09-29.

## System

- Manufacturer: HP
- Model: HP ZBook 8 G1i 16 inch Mobile Workstation PC
- System family: HP ZBook
- System type: x64-based PC
- Operating system: Microsoft Windows 11 Enterprise, 64-bit
- OS build: 10.0.26200 (build 26200)

## Processor

- CPU: Intel Core Ultra 7 265H
- Cores: 16
- Logical processors: 16
- Current/max reported clock: 2.20 GHz
- L2 cache: 30,720 KB
- L3 cache: 24,576 KB
- Virtualization firmware: disabled according to Windows WMI

## Memory

- Installed RAM: 32 GB nominal (33,751,646,208 bytes reported; approximately 31.4 GiB)
- Configuration: 2 x 16 GB Samsung modules
- Module part number: M425R2GA3EB0-CWMOD
- Configured speed: DDR5-5600
- Slots populated: Bottom-Slot 1 (top), Bottom-Slot 2 (under)

## Graphics

### Integrated GPU

- GPU: Intel Arc Pro 140T GPU (driver-reported name includes 16GB)
- Driver: 32.0.101.8831
- Current display mode: 1920 x 1200 at 60 Hz
- WMI-reported adapter RAM: approximately 2 GB
- Note: the adapter name's 16 GB label is not treated as confirmed dedicated VRAM; Intel integrated graphics can use shared system memory.

### Discrete GPU

- GPU: NVIDIA RTX 500 Ada Generation Laptop GPU
- VRAM: 4,094 MiB (approximately 4 GB)
- Driver: 595.95
- CUDA availability: confirmed working for this project in the local Python environment
- Idle reading during collection: P8, 55 C, 0% utilization, 3.51 W, 210 MHz graphics clock

## Storage

- Drive: SK hynix PC801 HFS512GEJ9X101N NVMe SSD
- Capacity: 512,105,932,800 bytes (approximately 512 GB decimal / 476.9 GiB)
- Interface reported by WMI: SCSI
- Health status: OK
- Windows volume: C: (NTFS, label WINDOWS)
- C: capacity: 510,335,643,648 bytes
- C: free space: 245,275,529,216 bytes (approximately 228.4 GiB)

## Display

- Internal panel detected as: Generic PnP Monitor, BOE device identifier
- Active resolution: 1920 x 1200
- Refresh rate: 60 Hz
- Logical pixel density: 120 DPI

## Firmware and Mainboard

- Motherboard: HP product 8D94
- Board version: KBC Version 52.31.00
- BIOS: HP X70 Ver. 01.05.01
- BIOS release date reported by Windows: 2026-07-13

## Battery

- Battery: Internal primary battery
- Chemistry code: 2 (lithium ion according to the Windows WMI enumeration)
- Design voltage: 16,593 mV
- Charge at collection: 91%
- Estimated runtime at collection: 80 minutes
- Battery status: OK, currently discharging
- Design/full-charge capacities were not exposed by this Windows WMI provider.

## Connectivity

- Wi-Fi: Intel Wi-Fi 7 BE201 320MHz
- Current Wi-Fi link speed reported by Windows: 400 Mbps
- Wired Ethernet: Intel Ethernet Connection (24) I219-LM
- Bluetooth: Intel Wireless Bluetooth
- Active network at collection: Wi-Fi

## Peripherals and Controllers

- Camera: HP 5MP Camera (Realtek)
- Audio: Realtek High Definition Audio
- Additional audio devices: Intel Smart Sound Technology for USB Audio, Digital Microphones, and Bluetooth Audio
- USB controllers: 2 USB4 host routers and 2 Intel USB 3.20 xHCI controllers

## Project-Relevant Notes

- The NVIDIA GPU is the available CUDA device for PyTorch workloads.
- The GPU has approximately 4 GB VRAM; whole-slide image inference can approach this limit.
- The project environment previously confirmed CUDA-enabled PyTorch with `torch.cuda.is_available()` returning `True` and the NVIDIA GPU selected successfully.

## Collection Notes

- Values came from Windows CIM/WMI queries and `nvidia-smi`.
- Dynamic values such as free disk space, battery charge, GPU temperature, utilization, and power draw are snapshots from the collection time.
- Hardware serial numbers, MAC addresses, IP addresses, and other device identifiers are intentionally omitted from this project note.
