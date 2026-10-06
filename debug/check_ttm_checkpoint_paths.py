from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
import torch

for path in ['ttm_finetuned_models/ttm_finetuned',
             'ttm_full_finetuned_models/ttm_finetuned']:
    try:
        m = TinyTimeMixerForPrediction.from_pretrained(path)
        m.eval()
        out = m(past_values=torch.randn(1, 52, 1),
                freq_token=torch.tensor([3]))
        print(f'{path}: OK, shape {tuple(out.prediction_outputs.shape)}')
    except Exception as e:
        print(f'{path}: FAILED - {type(e).__name__}: {e}')
