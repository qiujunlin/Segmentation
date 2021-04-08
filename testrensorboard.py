from tensorboardX import SummaryWriter
log_dir= "E:\workspace\python\CPFNet_Project\Log\OCT"
writer = SummaryWriter(log_dir=log_dir)
step = 0
dice = 0
acc = 0
for i  in range(100):
    loss =0
    for  j in range(100):
        step += 1
        loss+=0.01
        if step % 10 == 0:
            writer.add_scalar('Train/loss_step', loss, step)
    writer.add_scalar('Train/loss_epoch', float(loss), i)

    if(i%10==0):
        dice+=0.01
        acc+=0.02
        writer.add_scalar('Valid/Dice1_val', dice, i)
        writer.add_scalar('Valid/Acc_val', acc, i)
