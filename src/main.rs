use polars::prelude::*;
use ndarray::prelude::*;

fn read_csv(file_path: &str) -> PolarsResult<DataFrame> {
    CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some(file_path.into()))?
        .finish()
}


fn filter_all_ones(df: &DataFrame) -> PolarsResult<DataFrame> {
    let mask = df
    .column("Department")?
    .as_materialized_series()
    .equal("Cardiology")?;

    df.filter(&mask)
}

fn get_col_vector(df: &DataFrame, col_name: &str) -> PolarsResult<Vec<f64>> {
    let column = df.column(col_name)?.f64()?;
    Ok(column.into_no_null_iter().collect())
}



fn prepare_xy(x_vals: &Vec<f64>, y_vals: &Vec<f64>) -> (Array2<f64>, Array1<f64>) {
    // Turn x_vec, y_vec into ndarray arrays
    let x_array = Array1::from_vec(x_vals.clone());
    let y_array = Array1::from_vec(y_vals.clone());

    // Reshape x into N×1 matrix
    let binding = x_array.clone();
    let x_matrix = binding.to_shape((x_array.len(), 1)).unwrap();

    // Add intercept term (bias column)
    let ones = Array2::ones((x_matrix.nrows(), 1));
    let x = ndarray::concatenate![Axis(1), ones, x_matrix];

    (x, y_array)
}


fn cost(x: &Array2<f64>, y: &Array1<f64>, beta: &Array1<f64>) -> f64 {
    let predictions = x.dot(beta);
    let error = &predictions - y;
    let squared_error = (&error * &error).sum();
    let cost = squared_error / (2.0 * y.len() as f64);
    return cost;
}
fn gradient_descent(
    x: &Array2<f64>,
    y: &Array1<f64>,
    learning_rate: f64,
    iterations: usize,
) -> (Array1<f64>, Vec<f64>) {
    let mut beta = Array1::zeros(x.ncols());
    let mut cost_history: Vec<f64> = Vec::with_capacity(iterations);
    let m = y.len() as f64;

    for i in 0..iterations {
        let preds = x.dot(&beta);
        let error = &preds - y;
        let grad = x.t().dot(&error) / m;
        let update = &grad * learning_rate;
        beta = &beta - &update;

        let current_cost = cost(x, y, &beta);
        cost_history.push(current_cost);

        
        if (i + 1) % 1000000 == 0 {
           
            let beta_formatted: Vec<String> = beta.iter().map(|v| format!("{:.2}", v)).collect();
            println!(
                "Iteration {}: beta = [{},], cost = {:.4}",
                i + 1,
                beta_formatted.join(", "),
                current_cost
            );
        }
    }

    (beta, cost_history)
}


fn main() {
    let df = read_csv("src/data.csv").unwrap();

    // Filtering the dataframe
    let filtered_df = filter_all_ones(&df).unwrap();

 
    let x_vec = get_col_vector(&filtered_df, "TreatmentHours").unwrap();
    let y_vec = get_col_vector(&filtered_df, "RecoveryScore").unwrap();

    println!("x_vec : {:?}", x_vec);
    println!("y_vec : {:?}", y_vec);

    // Preparing X and y 
    let (x_matrix, y_array) = prepare_xy(&x_vec, &y_vec);

    // Hyperparameters
    let learning_rate = 0.0001;
    let iterations: usize = 1_000_000;

    // Time for the training (gradient descent)
    let start = std::time::Instant::now();
    let (beta, cost_history) = gradient_descent(&x_matrix, &y_array, learning_rate, iterations);
    let duration = start.elapsed();
    println!("\nTraining time: {:.4} seconds", duration.as_secs_f64());

    //  Final results 
    println!("\nFinal Parameters:");
    if beta.len() >= 2 {
        println!("beta0 (intercept): {:.2}", beta[0]);
        println!("beta1 (slope): {:.2}", beta[1]);
    } else if beta.len() == 1 {
        println!("beta0 (intercept): {:.2}", beta[0]);
        println!("beta1 (slope): N/A");
    } else {
        println!("beta0 (intercept): N/A");
        println!("beta1 (slope): N/A");
    }

    if let Some(last_cost) = cost_history.last() {
        println!("\nFinal Cost: {:.4}", last_cost);
    } else {
        println!("\nFinal Cost: N/A");
    }


}
