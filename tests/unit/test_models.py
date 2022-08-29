import torch
from recommender.BERT4Rec import BERT4Rec


def test_bert4rec():
    n_items = 1000

    model = BERT4Rec(vocab_size=1000)

    src_items = torch.randint(low=0, high=n_items, size=(32, 30))

    src_items[:, 0] = 1

    trg_out = torch.randint(low=0, high=n_items, size=(32, 30))

    out = model(src_items)

    assert out.shape == torch.Size([32, 30, 1000])

    loss = model.training_step((src_items, trg_out), batch_idx=1)

    assert isinstance(loss, torch.Tensor)

    assert not torch.isnan(loss).any()

    assert loss.size() == torch.Size([])
