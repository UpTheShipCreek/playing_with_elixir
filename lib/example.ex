defmodule Example do
  use Application
  alias VegaLite
  alias Tucan

  @tolerance 2
  @epsilon 1.0e-9
  @num_of_points_in_cluster 50
  @num_of_clusters 4
  @random_restarts 20
  @spread 0.5



  def start(_type, _args) do
    Example.main()
    Supervisor.start_link([], strategy: :one_for_one)
  end

  def main do
    cluster_1 = generate_cluster([-1, 0], @num_of_points_in_cluster)
    cluster_2 = generate_cluster([1, 0], @num_of_points_in_cluster)
    cluster_3 = generate_cluster([0, -1], @num_of_points_in_cluster)
    cluster_4 = generate_cluster([0, 1], @num_of_points_in_cluster)

    points = cluster_1 ++ cluster_2 ++ cluster_3 ++ cluster_4

    {assignments, _, _} = k_means(points, @num_of_clusters, @random_restarts)
    data =
      Enum.zip(points, assignments)
      |> Enum.map(fn {[x, y], cluster} ->
        %{
          "x" => x,
          "y" => y,
          "cluster" => "cluster_#{cluster}"
        }
      end)

    Tucan.scatter(data, "x", "y", color_by: "cluster", shape_by: "cluster", point_size: 100)
    |> Tucan.Export.save!("kmeans_plot.png", scale: 2)
  end

  def euclidean_distance(v1, v2) do
    Enum.reduce(Enum.zip(v1, v2), 0, fn {x1, x2}, acc -> acc + :math.pow(x1 - x2, 2) end)
    |> :math.sqrt()
  end

  def norm(v) do
    euclidean_distance(v, List.duplicate(0, length(v)))
  end

  def generate_random_point(center, spread \\ @spread) do
    Enum.map(center, fn c -> c + (:rand.uniform() * 2 - 1) * spread end)
  end

  def generate_cluster(center, n, spread \\ @spread) do
    Enum.map(1..n, fn _ -> generate_random_point(center, spread) end)
  end

  def variance([], _centroid), do: 0

  def variance(points, centroid) do
    Enum.reduce(
      points,
      0,
      fn point, acc -> acc + :math.pow(euclidean_distance(point, centroid), 2) end
    ) / length(points)
  end

  def assign_to_cluster(point, centroids) do
    Enum.with_index(centroids)
    |> Enum.min_by(fn {centroid, _index} -> euclidean_distance(point, centroid) end)
    |> elem(1)
  end

  def assign_to_cluster_all(points, centroids) do
    Enum.map(points, fn point -> assign_to_cluster(point, centroids) end)
  end

  def break_condition(old_centroids, new_centroids) do
    Enum.zip(old_centroids, new_centroids)
    |> Enum.reduce(true, fn {old, new}, acc -> acc
    and (euclidean_distance(old, new) / (((norm(old) + norm(new)) / 2) + @epsilon)) < @tolerance end)
  end

  def recalculate_centroid(points) do
    n = length(points)
    Enum.zip_with(points, fn coords -> Enum.sum(coords) / n end)
  end

  def recalculate_centroid_all(points, assignments) do
    clusters = Enum.group_by(Enum.zip(points, assignments), fn {_point, assignment} -> assignment end, fn {point, _assignment} -> point end)
    correct_order_clusters = Enum.map(0..(map_size(clusters)-1), fn i -> Map.get(clusters, i, []) end)
    Enum.map(correct_order_clusters, fn cluster_points -> recalculate_centroid(cluster_points) end)
  end

  def k_means_loop(points, centroids) do
    assignments = assign_to_cluster_all(points, centroids)
    new_centroids = recalculate_centroid_all(points, assignments)
    if break_condition(centroids, new_centroids) do
      {assignments, new_centroids}
    else
      k_means_loop(points, new_centroids)
    end
  end

  def k_means(points, k, restarts) do
    Enum.map(1..restarts, fn _ ->
      # Generate random centroids from points
      centroids = Enum.map(1..k, fn _ -> Enum.random(points) end)
      # Run the k-means loop
      {assignment, new_centroids} = k_means_loop(points, centroids)
      # Calculate the total variance by cluster
      clusters = Enum.group_by(Enum.zip(points, assignment), fn {_point, assignment} -> assignment end, fn {point, _assignment} -> point end)
      variances = Enum.map(0..(map_size(clusters)-1), fn i -> variance(Map.get(clusters, i, []), Enum.at(new_centroids, i)) end)
      # Sum the variances of each cluster
      total_variance = Enum.sum(variances)
      {assignment, new_centroids, total_variance}
    end)
    # Return the assignment and centroids with the lowest variance
    |> Enum.min_by(fn {_assignment, _centroids, variance} -> variance end)
  end
end
