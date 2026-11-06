"""MOSHI_STAGE_EVENTS_CAP semantics: cursor = len(x) then x[cursor:] / reversed(x[cursor:]) / .get() behave as with
an uncapped list while memory stays bounded."""
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("MOSHI_STAGE_EVENTS_CAP", "8")

if __name__ == "__main__":
    from vllm_neuron.vllm.platform import NeuronPlatform
    NeuronPlatform.set_device_count(4)
    from moshi_neuron.serving.phases import install_stage_events_cap
    from vllm_omni.metrics import stats as st
    ref_list = []
    install_stage_events_cap()
    a = st.OrchestratorAggregator(num_stages=2, log_stats=False, wall_start_ts=0.0, final_stage_id_for_e2e=1)
    ok = True
    for i in range(1000):
        cur = len(a.stage_events.get("r", ()))
        a.stage_events.setdefault("r", []).append(i)
        ref_list.append(i)
        got = a.stage_events.get("r", [])[cur:]
        exp = ref_list[cur:]
        if list(got) != exp or list(reversed(a.stage_events["r"][cur:])) != list(reversed(exp)):
            ok = False
            print("mismatch at", i, got, exp)
            break
    print("cap semantics ok:", ok, "stored", list.__len__(a.stage_events["r"]), "logical", len(a.stage_events["r"]))
