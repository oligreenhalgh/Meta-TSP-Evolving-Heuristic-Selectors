use rand::seq::SliceRandom;
use rand::Rng;
use rayon::prelude::*;

#[derive(Clone)]
struct City {
    x: f64,
    y: f64,
}

struct TspInstance {
    cities: Vec<City>,
    dist_matrix: Vec<Vec<f64>>,
}

#[derive(Clone)]
struct Individual {
    tour: Vec<usize>,
    distance: f64,
}

impl TspInstance {
    fn new(cities: Vec<City>) -> Self {
        let n = cities.len();
        let mut dist_matrix = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                let d = ((cities[i].x - cities[j].x).powi(2) +
                    (cities[i].y - cities[j].y).powi(2)).sqrt();
                dist_matrix[i][j] = d;
            }
        }
        TspInstance { cities, dist_matrix }
    }

    fn calculate_total_distance(&self, tour: &[usize]) -> f64 {
        tour.windows(2)
            .map(|w| self.dist_matrix[w[0]][w[1]])
            .sum::<f64>() + self.dist_matrix[tour[tour.len()-1]][tour[0]]
    }

    pub fn simulated_annealing(&self, initial_temp: f64, cooling_rate: f64) -> Individual {
        let mut rng = rand::rng();
        let n = self.cities.len();
        let mut current_tour: Vec<usize> = (0..n).collect();
        current_tour.shuffle(&mut rng);
        let mut current_dist = self.calculate_total_distance(&current_tour);

        let mut best_tour = current_tour.clone();
        let mut best_dist = current_dist;
        let mut temp = initial_temp;

        for _ in 0..10000 {
            // Select a segment to reverse
            let mut i = rng.random_range(0..n);
            let mut j = rng.random_range(0..n);
            if i > j { std::mem::swap(&mut i, &mut j); }

            if i != j {
                // Perform the 2-opt move
                current_tour[i..=j].reverse();

                let new_dist = self.calculate_total_distance(&current_tour);
                let delta = new_dist - current_dist;

                // Metropolis Criterion: Accept if better, or with probability based on temperature
                if delta < 0.0 || rng.random_bool((-delta / temp).exp().min(1.0)) {
                    current_dist = new_dist;
                    if current_dist < best_dist {
                        best_dist = current_dist;
                        best_tour = current_tour.clone();
                    }
                } else {
                    // Reject the move: reverse the segment back
                    current_tour[i..=j].reverse();
                }
            }
            current_tour.swap(i, j);
            let new_dist = self.calculate_total_distance(&current_tour);
            let delta = new_dist - current_dist;

            if delta < 0.0 || rng.random_bool((-delta / temp).exp().min(1.0)) {
                current_dist = new_dist;
                if current_dist < best_dist {
                    best_dist = current_dist;
                    best_tour = current_tour.clone();
                }
            } else {
                current_tour.swap(i, j);
            }
            temp *= cooling_rate;
            if temp < 0.01 { break; }
        }
        Individual { tour: best_tour, distance: best_dist }
    }

    pub fn two_opt(&self, mut tour: Vec<usize>) -> Individual {
        let n = tour.len();
        let mut improved = true;

        while improved {
            improved = false;
            for i in 1..n - 1 {
                for j in i + 1..n {
                    let d_old = self.dist_matrix[tour[i-1]][tour[i]] + self.dist_matrix[tour[j % n]][tour[(j+1) % n]];
                    let d_new = self.dist_matrix[tour[i-1]][tour[j % n]] + self.dist_matrix[tour[i]][tour[(j+1) % n]];

                    if d_new < d_old {
                        tour[i..=j].reverse();
                        improved = true;
                    }
                }
            }
        }
        let distance = self.calculate_total_distance(&tour);
        Individual { tour, distance }
    }
}

#[derive(Clone)]
struct Selector {
    weights: [f64; 3],
}

impl Selector {
    fn predict(&self, n: usize, density: f64) -> usize {
        let feat_n = n as f64 / 600.0;
        let feat_d = density * 10.0;
        let score = (feat_n * self.weights[0]) + (feat_d * self.weights[1]) + self.weights[2];
        if score > 0.0 { 1 } else { 0 }
    }
}

fn main() {
    let mut rng = rand::rng();
    let mut population: Vec<Selector> = (0..20).map(|_| Selector {
        weights: [rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0), rng.random_range(-1.0..1.0)]
    }).collect();

    for g_idx in 0..50 {
        let scores: Vec<(f64, usize)> = population.par_iter().enumerate().map(|(p_idx, selector)| {
            let mut total_regret = 0.0;
            let scenarios = [(50, 2), (200, 5), (600, 10)];

            for (n, clusters) in scenarios {
                let instance = TspAdversary::generate_clustered(n, clusters);
                let density = clusters as f64 / n as f64;

                let mut local_rng = rand::rng();
                let mut start_tour: Vec<usize> = (0..n).collect();
                start_tour.shuffle(&mut local_rng);

                // 1. Run Benchmarks
                // We use .clone() on start_tour so 'two_opt' doesn't consume the only copy
                let sa_result = instance.simulated_annealing(100.0, 0.999);
                let opt_result = instance.two_opt(start_tour.clone());

                // 2. Selector Makes a Prediction
                let choice = selector.predict(n, density);
                let chosen_dist = if choice == 1 { opt_result.distance } else { sa_result.distance };

                // 3. Calculate Regret
                let best_dist = sa_result.distance.min(opt_result.distance);
                total_regret += (chosen_dist - best_dist) / best_dist;
            }
            (-total_regret, p_idx)
        }).collect();

        // Sorting & Reproduction Logic
        let mut sorted = scores.clone();
        sorted.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());


        let avg_pop_regret: f64 = scores.iter().map(|s| -s.0).sum::<f64>() / scores.len() as f64;
        println!("Gen {:>2} | Best Regret: {:.4} | Avg Regret: {:.4}", g_idx, -sorted[0].0, avg_pop_regret);


        let mut next_gen = Vec::with_capacity(20);
        for i in 0..5 {
            let parent = &population[sorted[i].1];
            next_gen.push(parent.clone()); // Elitism

            for _ in 0..3 {
                let mut new_weights = parent.weights;
                let mut m_rng = rand::rng();
                for w in new_weights.iter_mut() {
                    *w += m_rng.random_range(-0.1..0.1);
                }
                next_gen.push(Selector { weights: new_weights });
            }
        }
        population = next_gen;
    }
}

struct TspAdversary;
impl TspAdversary {
    fn generate_clustered(n: usize, clusters: usize) -> TspInstance {
        let mut rng = rand::rng();
        let mut cities = Vec::new();
        let centers: Vec<(f64, f64)> = (0..clusters)
            .map(|_| (rng.random_range(0.0..100.0), rng.random_range(0.0..100.0)))
            .collect();
        for i in 0..n {
            let center = centers[i % clusters];
            cities.push(City {
                x: center.0 + rng.random_range(-5.0..5.0),
                y: center.1 + rng.random_range(-5.0..5.0),
            });
        }
        TspInstance::new(cities)
    }
}