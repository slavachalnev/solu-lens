import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader
# from datasets import load_dataset
import transformer_lens
from transformer_lens import HookedTransformer
from transformer_lens import utils

from mlp import SoluMLP, GeluMLP



d_model = 768
solu_layer = SoluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)
big_solu_layer = SoluMLP(input_size=d_model, hidden_size=d_model*8, output_size=d_model)
gelu_layer = GeluMLP(input_size=d_model, hidden_size=d_model*4, output_size=d_model)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

gpt2 = HookedTransformer.from_pretrained("gpt2-small", device=device)
tokenizer = gpt2.tokenizer

loader = transformer_lens.evals.make_pile_data_loader(tokenizer=tokenizer, batch_size=8)


pre = None
post = None

def mlp_pre(value, hook):
    global pre
    pre = value.detach().clone().cpu()
    return value

def mlp_post(value, hook):
    global post
    post = value.detach().clone().cpu()
    return value

layer_to_hook = 9

fwd_hooks = [
    (utils.get_act_name("attn_out", layer_to_hook), mlp_pre),
    (utils.get_act_name("mlp_out", layer_to_hook), mlp_post)
]

solu_layer.to(device)
big_solu_layer.to(device)
gelu_layer.to(device)

solu_layer.train()
big_solu_layer.train()
gelu_layer.train()

criterion = nn.MSELoss()
s_optimizer = torch.optim.Adam(solu_layer.parameters(), lr=1e-3)
bs_optimizer = torch.optim.Adam(big_solu_layer.parameters(), lr=1e-3)
g_optimizer = torch.optim.Adam(gelu_layer.parameters(), lr=1e-3)

writer = SummaryWriter()

for epoch in range(10):
    s_loss_total = 0
    bs_loss_total = 0
    g_loss_total = 0
    for idx, batch in enumerate(loader):
        # print("Batch")
        # print(batch)
        with torch.no_grad():
            loss = gpt2.run_with_hooks(
                batch["tokens"].to(device), 
                return_type="loss", 
                fwd_hooks=fwd_hooks,
            )
        s_optimizer.zero_grad()
        bs_optimizer.zero_grad()
        g_optimizer.zero_grad()

        s_loss = criterion(solu_layer(pre.to(device)), post.to(device))
        bs_loss = criterion(big_solu_layer(pre.to(device)), post.to(device))
        g_loss = criterion(gelu_layer(pre.to(device)), post.to(device))

        s_loss.backward()
        bs_loss.backward()
        g_loss.backward()

        s_optimizer.step()
        bs_optimizer.step()
        g_optimizer.step()

        s_loss_total += s_loss.item()
        bs_loss_total += bs_loss.item()
        g_loss_total += g_loss.item()

        if idx % 10 == 0:
            # log to tensorboard
            writer.add_scalar("Loss/S", s_loss.item(), epoch*len(loader) + idx)
            writer.add_scalar("Loss/G", g_loss.item(), epoch*len(loader) + idx)
            writer.add_scalar("Loss/BS", bs_loss.item(), epoch*len(loader) + idx)


        print(f"Epoch: {epoch} step {idx}, S Loss: {s_loss}, BS Loss: {bs_loss}, G Loss: {g_loss}")
    print(f"end Epoch: {epoch}, S Loss: {s_loss_total/len(loader)}, BS Loss: {bs_loss_total/len(loader)}, G Loss: {g_loss_total/len(loader)}")
    print('saving')

    torch.save(solu_layer.state_dict(), f"solu_{epoch}.pt")
    torch.save(big_solu_layer.state_dict(), f"big_solu_{epoch}.pt")
    torch.save(gelu_layer.state_dict(), f"gelu_{epoch}.pt")

