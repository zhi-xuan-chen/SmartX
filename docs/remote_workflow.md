# Remote Workflow

Work on this repository from `jhcpu7`:

```bash
cd ~/Workspace/SmartX
source ~/miniconda3/etc/profile.d/conda.sh
conda activate vlp
```

Preferred connection route when direct SSH to `ras.cse.ust.hk` is unreliable:

```text
local -> ssh -i ~/.ssh/zhefan.pem ec2-user@44.255.75.53 -> ras.cse.ust.hk -> jhcpu7.cse.ust.hk
```

Keep the SSH session open during an active review/edit session. Do not disconnect after each small task unless the user asks, the connection hangs, or there is an operational risk.

Do not write plaintext SSH passwords, tokens, or private credentials into repository files. Use credentials only from the active conversation or a secure credential source.
