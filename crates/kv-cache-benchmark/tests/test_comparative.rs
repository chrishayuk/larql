use kv_cache_benchmark::benchmark;
use kv_cache_benchmark::graph_walk::GraphWalk;
use kv_cache_benchmark::markov_residual::MarkovResidual;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::rotorquant::RotorQuantStrategy;
use kv_cache_benchmark::standard_kv::StandardKv;
use kv_cache_benchmark::turboquant::TurboQuant;
use kv_cache_benchmark::*;

#[test]
fn test_all_strategies_memory_ordering() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    for &seq_len in &[4096, 32768, 370_000] {
        let mem_std = standard.memory_bytes(&config, seq_len);
        let mem_tq = tq4.memory_bytes(&config, seq_len);
        let mem_mrk = markov.memory_bytes(&config, seq_len);
        let mem_gw = graph.memory_bytes(seq_len);

        // Standard KV is always the largest.
        assert!(
            mem_std > mem_tq,
            "At {seq_len}: Standard ({mem_std}) > TurboQuant ({mem_tq})"
        );

        // MarkovRS W=512 is bounded by the hot window (~192 MB) regardless of seq_len.
        // At short contexts (<~11K) the window dominates and MarkovRS > TurboQuant.
        // At long contexts TurboQuant grows larger. Both beat standard KV.
        assert!(
            mem_std > mem_mrk,
            "At {seq_len}: Standard ({mem_std}) > Markov RS ({mem_mrk})"
        );

        // Graph Walk is the per-conversation minimum (token IDs only).
        assert!(
            mem_gw < mem_mrk,
            "At {seq_len}: Graph Walk ({mem_gw}) < Markov RS ({mem_mrk})"
        );
    }

    // At very long contexts, MarkovRS stays flat while TurboQuant grows O(n).
    // Crossover: MarkovRS fixed window (~192 MB) < TurboQuant at ~11K+ tokens.
    let mem_mrk_370k = markov.memory_bytes(&config, 370_000) as f64;
    let mem_tq_370k = tq4.memory_bytes(&config, 370_000) as f64;
    assert!(
        mem_tq_370k > mem_mrk_370k,
        "At 370K: TurboQuant ({mem_tq_370k:.0}) should exceed Markov RS ({mem_mrk_370k:.0})"
    );
}

#[test]
fn test_memory_sweep_produces_data() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);

    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov];
    let lengths = &[512, 4096, 32768];

    let points = benchmark::memory_sweep(&config, &strategies, lengths);

    // 3 strategies × 3 lengths = 9 points
    assert_eq!(points.len(), 9);

    for point in &points {
        assert!(
            point.memory_bytes > 0,
            "Zero memory for {}",
            point.strategy_name
        );
    }
}

#[test]
fn test_comparative_table_format() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let rq_iso3 = RotorQuantStrategy::iso3();
    let rq_planar3 = RotorQuantStrategy::planar3();

    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov, &rq_iso3, &rq_planar3];
    let table = benchmark::format_comparative_table(&config, &strategies);

    assert!(table.contains("Gemma 3-4B"));
    assert!(table.contains("Standard KV"));
    assert!(table.contains("TurboQuant"));
    assert!(table.contains("Markov Residual Stream"));
    assert!(table.contains("RotorQuant Iso3"));
    assert!(table.contains("RotorQuant Planar3"));
}

#[test]
fn test_370k_memory_ratios() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    let seq_len = 370_000;
    let mem_std = standard.memory_bytes(&config, seq_len) as f64;
    let mem_tq = tq4.memory_bytes(&config, seq_len) as f64;
    let mem_mrk = markov.memory_bytes(&config, seq_len) as f64;
    let mem_gw = graph.memory_bytes(seq_len) as f64;

    let ratio_tq = mem_std / mem_tq;
    let ratio_mrk = mem_std / mem_mrk;
    let ratio_gw = mem_std / mem_gw;

    // TurboQuant: 4-6× compression
    assert!(ratio_tq > 2.0, "TQ ratio: {ratio_tq:.1}×");
    assert!(ratio_tq < 8.0, "TQ ratio: {ratio_tq:.1}×");

    // Markov RS: 100×+ compression
    assert!(ratio_mrk > 100.0, "Markov ratio: {ratio_mrk:.1}×");

    // Graph Walk: per-conversation is even smaller (token IDs only).
    assert!(
        ratio_gw > ratio_mrk,
        "Graph Walk should compress more than Markov RS"
    );

    println!("At 370K tokens on {}:", config.name);
    println!("  Standard KV:   {:.1} GB", mem_std / 1e9);
    println!("  TurboQuant 4b: {:.1} GB ({ratio_tq:.1}×)", mem_tq / 1e9);
    println!("  Markov RS:     {:.1} MB ({ratio_mrk:.0}×)", mem_mrk / 1e6);
    println!("  Graph Walk:    {:.1} MB ({ratio_gw:.0}×)", mem_gw / 1e6);
}

#[test]
fn test_multi_model_memory() {
    let models = ModelConfig::all();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);

    for config in &models {
        let std_4k = standard.memory_bytes(config, 4096);
        let tq_4k = tq4.memory_bytes(config, 4096);
        assert!(
            std_4k > tq_4k,
            "{}: Standard ({std_4k}) should > TurboQuant ({tq_4k})",
            config.name
        );
    }
}
