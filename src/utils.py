

def run_step(batch, net, tokenizer, loss_name, device):
    e1_inputs = tokenizer(text=batch['full_story'], 
                        text_pair=batch['ending1'],
                        padding=True,
                        truncation=True,
                        max_length=128,
                        return_tensors="pt",)
    e2_inputs = tokenizer(text=batch['full_story'], 
                    text_pair=batch['ending2'],
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt")

    for i in e1_inputs.keys():
        e1_inputs[i]=e1_inputs[i].to(device)
        e2_inputs[i]=e2_inputs[i].to(device)

    output = net(e1_inputs, e2_inputs)
    loss = loss_name(output, batch['labels'])

    return output, loss

