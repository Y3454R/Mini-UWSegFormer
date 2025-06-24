```
Input Image
    ↓
  Encoder (MiT-B0)
    ↓
[UIQA] → enhances feature quality
    ↓
[MAA] → merges multi-scale features
    ↓
Decoder Head
    ↓
Output Mask
    ↓
[ELL] → used during training to refine edges
```
