for (( i=1; i<=5; i++ ))
do
    echo "Training Model: $i"
    python simulate.py --model dual_ppo --patient_num $i
    echo ""
done
