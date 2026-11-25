from gtda.diagrams import PersistenceEntropy
from gtda.homology import VietorisRipsPersistence
from gtda.plotting import plot_diagram
from gtda.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from gtda.plotting import plot_point_cloud

# Track connected components, loops, and voids
homology_dimensions = [0, 1, 2]

# Collapse edges to speed up H2 persistence calculation!
persistence = VietorisRipsPersistence(
    metric="euclidean",
    homology_dimensions=homology_dimensions,
    n_jobs=6,
    collapse_edges=True,
)
persistence_entropy = PersistenceEntropy(normalize=True)

#TODO: input-point_clouds
persistence_diagrams = persistence.fit_transform(point_clouds)
# Circle
plot_diagram(persistence_diagrams[0])
# Sphere
plot_diagram(persistence_diagrams[10])
# Torus
plot_diagram(persistence_diagrams[-1])
# Index - (human_arms_out, 0), (vase, 10), (dining_chair, 20), (biplane, 30)
index = 30
plot_diagram(persistence_diagrams[index])


#TODO: persistent entropy矢量化
## calculate topological feature matrix
## X_basic--expect shape - (n_point_clouds, n_homology_dims)
X_basic = persistence_entropy.fit_transform(persistence_diagrams)

# Visualise feature matrix
plot_point_cloud(X_basic)


steps = [
    ("persistence", VietorisRipsPersistence(metric="euclidean", homology_dimensions=homology_dimensions, n_jobs=6)),
    ("entropy", PersistenceEntropy()),
    ("model", RandomForestClassifier(oob_score=True)),
]
pipeline = Pipeline(steps)
pipeline.fit(point_clouds_basic, labels_basic)