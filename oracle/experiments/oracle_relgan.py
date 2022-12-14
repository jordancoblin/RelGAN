import os
from subprocess import call
import sys, time

# Job id and gpu_id
if len(sys.argv) > 2:
    job_id = int(sys.argv[1])
    gpu_id = str(sys.argv[2])
    print('job_id: {}, gpu_id: {}'.format(job_id, gpu_id))
elif len(sys.argv) > 1:
    job_id = int(sys.argv[1])
    gpu_id = '0'
    print('job_id: {}, missing gpu_id (use default {})'.format(job_id, gpu_id))
else:
    print('Missing argument: job_id and gpu_id.')
    quit()

# Executables
executable = 'python3'

# Arguments
# architecture = ['rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla', 'rmc_vanilla']
# gantype =      ['RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN', 'RSGAN']
# opt_type =     ['adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam', 'adam']
# temperature =  ['1', '1', '1', '1', '1', '1', '1', '1']
# lam =          [' 0.', ' -1.', ' -2.', ' -4.', ' -8.', '0.', '0.', '0.'] # Note the space prefix is needed to parse negative floats
# d_lr =         ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']
# gadv_lr =      ['1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4', '1e-4']
# mem_slots =    ['1', '1', '1', '1', '1', '1', '1', '1']
# head_size =    ['256', '256', '256', '256', '256', '256', '256', '256']
# num_heads =    ['2', '2', '2', '2', '2', '2', '2', '2']
seed =         ['99', '100', '101', '99', '100', '101', '99', '100', '101',' 99', '100', '101', '99', '100', '101',]
exp_type =     ['entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas', 'entmax_alphas']
alpha =        ['1.1', '1.1', '1.1', '1.3', '1.3', '1.3', '1.5', '1.5', '1.5', '1.7', '1.7', '1.7', '2.0', '2.0', '2.0']

architecture = 'rmc_vanilla'
gantype = 'RSGAN'
opt_type = 'adam'
# bs = '1'
bs = '64'
gpre_lr = '1e-2'
hidden_dim = '32'
seq_len = '20'
dataset = 'oracle'
vocab_size = '5000'

gsteps = '1'
dsteps = '5'
d_lr = '1e-4'
gadv_lr = '1e-4'
mem_slots = '1'
head_size = '256'
num_heads = '2'
gen_emb_dim = '32'
dis_emb_dim = '64'
num_rep = '64'
sn = False
decay = False
# adapt = 'exp'
adapt = 'no'
npre_epochs = '150'
nadv_steps = '5000'
ntest = '20'

sparse = True
temperature = '1'

# Paths
rootdir = '../..'
scriptname = 'run.py'
cwd = os.path.dirname(os.path.abspath(__file__))

outdir = os.path.join(cwd, 'out', exp_type[job_id], time.strftime("%Y%m%d"), dataset,
                      'oracle_{}_{}_{}_bs{}_sl{}_sn{}_dec{}_ad-{}_npre{}_nadv{}_ms{}_hs{}_nh{}_ds{}_dlr{}_glr{}_tem{}_demb{}_nrep{}_hdim{}_sd{}_alpha{}'.
                      format(architecture, gantype, opt_type, bs, seq_len, int(sn),
                             int(decay), adapt, npre_epochs, nadv_steps, mem_slots, head_size,
                             num_heads, dsteps, d_lr, gadv_lr, temperature,
                             dis_emb_dim, num_rep, hidden_dim, seed[job_id], alpha[job_id]))

args = [
    # Architecture
    '--gf-dim', '64',
    '--df-dim', '64',
    '--g-architecture', architecture,
    '--d-architecture', architecture,
    '--gan-type', gantype,
    '--hidden-dim', hidden_dim,

    # Training
    '--gsteps', gsteps,
    '--dsteps', dsteps,
    '--npre-epochs', npre_epochs,
    '--nadv-steps', nadv_steps,
    '--ntest', ntest,
    '--d-lr', d_lr,
    '--gpre-lr', gpre_lr,
    '--gadv-lr', gadv_lr,
    '--batch-size', bs,
    '--log-dir', os.path.join(outdir, 'tf_logs'),
    '--sample-dir', os.path.join(outdir, 'samples'),
    '--optimizer', opt_type,
    '--seed', seed[job_id],
    '--temperature', temperature,
    '--alpha', alpha[job_id],
    '--adapt', adapt,

    # evaluation
    '--nll-oracle',
    '--nll-gen',
    # '--doc-embsim',

    # relational memory
    '--mem-slots', mem_slots,
    '--head-size', head_size,
    '--num-heads', num_heads,

    # dataset
    '--dataset', dataset,
    '--vocab-size', vocab_size,
    '--start-token', '0',
    '--seq-len', seq_len,
    '--num-sentences', '10000',
    '--gen-emb-dim', gen_emb_dim,
    '--dis-emb-dim', dis_emb_dim,
    '--num-rep', num_rep,
    '--data-dir', './data']

if sn:
    args += ['--sn']
if decay:
    args += ['--decay']

# Run
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
my_env = os.environ.copy()
call([executable, scriptname] + args, env=my_env, cwd=rootdir)
