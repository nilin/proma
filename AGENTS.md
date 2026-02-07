in dp_actor.py proma part, add a new option proma_intra (bool) and a parameter proma_intra_dim default to 30 and another parameter proma_intra_use_same=False. this part will pick a
random subset of the positions for activations and a random subset for grad_out. if proma_intra_use_same it's the same subset. then we project out g_i a_i^t. if use_same then these are
the same position else theyre samples independently. oh and there's another option which chooses to project them out from either the accumulated grad or from the current microbatch grad.
everything should be able to do with vector dot products and u.T A v products, at least the batched operations across the 30 samples. but it might be clean to use a entrywise product of
matrices after. keep the changes minimal, but add the options to the actor.py settings file and add a function to run/run.sh
