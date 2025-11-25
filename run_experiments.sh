 #!/bin/bash

# Define number of runs per patient
RUNS=100
OUTPUT_CSV="final_results_with_deltas.csv"

# 1. Initialize CSV Header
# We interleave Average and Standard Deviation columns for the tracked states
echo "Patient_ID,Avg_Reward,Std_Reward,Avg_Steps,Avg_Steps_Range,Avg_Pct_Range,Std_Pct_Range,Avg_Pct_Hypo,Std_Pct_Hypo,Avg_Pct_Hyper,Std_Pct_Hyper,Avg_Pct_Sev_Hypo,Std_Pct_Sev_Hypo,Avg_Pct_Sev_Hyper,Std_Pct_Sev_Hyper" > "$OUTPUT_CSV"

echo "Starting Simulation Batch (Calculating Means and Std Devs)..."
echo "Results will be saved to: $OUTPUT_CSV"
echo "-----------------------------------"

for x in {1..10}; do
    echo "Processing Patient ID: $x"

    TEMP_FILE="temp_data_patient_${x}.txt"
    > "$TEMP_FILE"

    for (( i=1; i<=RUNS; i++ )); do
        echo -ne "  Running simulation $i/$RUNS ...\r"

        # Run simulation
        python simulate.py --model dual_ppo --patient_num "$x" > /dev/null 2>&1

        # Read output
        STATS_FILE="stats/stats${x}.txt"
        if [[ -f "$STATS_FILE" ]]; then
            cat "$STATS_FILE" >> "$TEMP_FILE"
            echo "" >> "$TEMP_FILE"
        else
            echo "  Warning: $STATS_FILE not found for run $i"
        fi
    done

    echo -e "\n  Computing statistics..."

    # 2. Calculate Mean AND Standard Deviation
    # Logic: Variance = (SumSq - (Sum^2 / N)) / (N - 1)
    awk -v N=$RUNS -v PATIENT_ID=$x '
        # Function to calculate Sample Standard Deviation
        function std(sum, sum_sq) {
            if (N < 2) return 0;
            variance = (sum_sq - (sum * sum / N)) / (N - 1);
            if (variance < 0) variance = 0;
            return sqrt(variance);
        }

        # Accumulate Sums and Sums of Squares
        /^tot_reward:/                     { s_rew += $2;   ss_rew += $2*$2 }
        /^num_steps:/                      { s_st += $2 }   # Usually don t need std for fixed steps
        /^steps_in_range:/                 { s_str += $2 }
        /^percent_in_range:/               { s_pr += $2;    ss_pr += $2*$2 }
        /^percent_in_hypoglycemic:/        { s_pho += $2;   ss_pho += $2*$2 }
        /^percent_in_hyperglycemic:/       { s_phy += $2;   ss_phy += $2*$2 }
        /^percent_in_severe_hypoglycemic:/ { s_psho += $2;  ss_psho += $2*$2 }
        /^percent_in_severe_hyperglycemic:/{ s_pshy += $2;  ss_pshy += $2*$2 }

        END {
            # Output formatted CSV line
            printf "%d,%.5f,%.5f,%.2f,%.2f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n",
            PATIENT_ID,
            s_rew/N,   std(s_rew, ss_rew),
            s_st/N,
            s_str/N,
            s_pr/N,    std(s_pr, ss_pr),
            s_pho/N,   std(s_pho, ss_pho),
            s_phy/N,   std(s_phy, ss_phy),
            s_psho/N,  std(s_psho, ss_psho),
            s_pshy/N,  std(s_pshy, ss_pshy)
        }
    ' "$TEMP_FILE" >> "$OUTPUT_CSV"

    rm "$TEMP_FILE"
    echo "-----------------------------------"

done

echo "Batch processing complete."
