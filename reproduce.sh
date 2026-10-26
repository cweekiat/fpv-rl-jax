#!/usr/bin/env bash
# Every training run and evaluation behind the results, one after another, on a laptop CPU.
#   short range: teacher, DAgger student, ablations                          (~3.5 h)
#   long range:  teachers, then the camera-only 2x2 (memory x RL fine-tuning)
#                with compute-matched DAgger controls, 5 seeds              (~2 days)
# Each step is skipped if its checkpoint already exists.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p ckpt/main logs/main
run() { local out=$1 name=$2; shift 2; [ -f "$out" ] && return; echo "== $name $(date +%H:%M)"
        uv run python train.py "$@" --out "$out" 2>&1 | tee "logs/$name.txt"; }

# ---- short range (targets 4-12 m)
run ckpt/teacher.pkl         teacher        teacher
run ckpt/student.pkl         student        student --teacher ckpt/teacher.pkl
run ckpt/student_1frame.pkl  student_1frame student --teacher ckpt/teacher.pkl --frame-stack 1
run ckpt/student_bc.pkl      student_bc     student --teacher ckpt/teacher.pkl --bc
run ckpt/teacher_nodr.pkl    teacher_nodr   teacher --dr 0 --iters 300
run ckpt/teacher_noperc.pkl  teacher_noperc teacher --w-perception 0 --iters 300
run ckpt/student_noperc.pkl  student_noperc student --teacher ckpt/teacher_noperc.pkl --w-perception 0

# ---- long range (targets 100-200 m, ~40 m/s drone): all-knowing, then visibility-limited teacher
LONG="--episode-seconds 30 --sag-floor 0.7 --level0 1.0"
run ckpt/lr_teacher.pkl      lr_teacher     teacher --preset long --iters 1500
run ckpt/lr_teacher_vis.pkl  lr_teacher_vis teacher --preset long --vis-teacher --init ckpt/lr_teacher.pkl --iters 200 $LONG

# ---- camera-only students: A = no memory, B = memory (GRU); imitation -> RL fine-tune or more DAgger
PROTO="--preset long --hist-len 10 --vis-teacher $LONG --level-mix --teacher ckpt/lr_teacher_vis.pkl \
       --beta-iters 250 --buffer-iters 12"
for s in 0 1 2 3 4; do
  for c in A B; do
    mem=""; [ $c = B ] && mem="--recurrent"
    run ckpt/main/${c}_s$s.pkl   main/${c}_s$s   student $PROTO $mem --iters 600 --seed $s
    run ckpt/main/${c}rl_s$s.pkl main/${c}rl_s$s finetune --init-student ckpt/main/${c}_s$s.pkl \
        --teacher ckpt/lr_teacher_vis.pkl --iters 200 --seed $s
    # compute-matched control: the same 1.6M steps as RL, as DAgger with the teacher no longer acting
    run ckpt/main/${c}dg_s$s.pkl main/${c}dg_s$s student $PROTO $mem --iters 100 --beta-iters 0 --lr 3e-5 \
        --init-student ckpt/main/${c}_s$s.pkl --seed $(( s + 100 ))
  done
done

# ---- evaluation (1024 episodes per pilot, same seeds for every pilot)
uv run python evaluate.py main
uv run python evaluate.py sweep
uv run python evaluate.py ablations
uv run python evaluate.py long
uv run python scripts/eval_long.py teachers ckpt/lr_teacher.pkl ckpt/lr_teacher_vis.pkl --episodes=1024
uv run python scripts/eval_long.py camera_main ckpt/main/{A,Arl,B,Brl}_s{0,1,2}.pkl --episodes=1024
uv run python scripts/eval_long.py seeds34 ckpt/main/{A,Arl,B,Brl,Adg,Bdg}_s{3,4}.pkl --episodes=1024
uv run python scripts/eval_long.py controls ckpt/main/{A,B}dg_s{0,1,2}.pkl --episodes=1024
uv run python scripts/visual_servo.py
for s in 0 1 2 3 4; do for c in A B Brl; do uv run python scripts/probe_memory.py ckpt/main/${c}_s$s.pkl; done; done
for s in 0 1 2; do uv run python scripts/probe_memory.py ckpt/main/Arl_s$s.pkl; done

# ---- figures and video
uv run python make_figures.py
uv run --with "imageio[ffmpeg]" python scripts/make_video.py
