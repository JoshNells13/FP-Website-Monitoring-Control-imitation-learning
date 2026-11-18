from app.routes.analytics import analytics_bp
from app.routes.checkpoints import checkpoints_bp
from app.routes.dashboard import dashboard_bp
from app.routes.experiments import experiments_bp
from app.routes.simulation import simulation_bp
from app.routes.training import training_bp
from app.routes.videos import videos_bp

ALL_BLUEPRINTS = [
    dashboard_bp,
    training_bp,
    simulation_bp,
    analytics_bp,
    experiments_bp,
    videos_bp,
    checkpoints_bp,
]
