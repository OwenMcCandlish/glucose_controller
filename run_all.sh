for (( i=1; i<=10; i++ ))
do
    echo "Training Model: $i"
    python simulate.py --model single_ppo --train --patient_num $i
    echo ""
done
