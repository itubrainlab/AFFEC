"""
Experiment logging and tracking
"""
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional


class ExperimentLogger:
    """
    Log experiments aligned with docs/knowledge-management.md
    
    Each experiment captures:
    - Configuration (data version, modality settings)
    - Results (metrics, model performance)
    - Decision rationale
    - Risks and follow-up actions
    """
    
    def __init__(self, log_dir: str = "logs", experiment_name: Optional[str] = None):
        """
        Initialize experiment logger
        
        Parameters:
        -----------
        log_dir : str
            Directory to store experiment logs
        experiment_name : str
            Name for this experiment (default: timestamp-based)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.experiment_name = experiment_name
        self.experiment_file = self.log_dir / f"{experiment_name}.json"
        self.experiment = {
            'timestamp': datetime.now().isoformat(),
            'name': experiment_name,
            'config': None,
            'results': None,
            'decisions': [],
            'risks': [],
            'next_actions': []
        }
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Log configuration (data version, modality settings, hyperparameters)
        
        Parameters:
        -----------
        config : dict
            Configuration dictionary
        """
        self.experiment['config'] = config
    
    def log_results(self, results: Dict[str, Any]) -> None:
        """
        Log results (metrics, model performance, per-fold details)
        
        Parameters:
        -----------
        results : dict
            Results dictionary
        """
        self.experiment['results'] = results
    
    def add_decision(self, title: str, rationale: str, evidence: Optional[str] = None) -> None:
        """
        Log a decision with rationale and evidence
        
        Parameters:
        -----------
        title : str
            Decision title
        rationale : str
            Why this decision was made
        evidence : str
            Data/evidence supporting the decision
        """
        decision = {
            'title': title,
            'rationale': rationale,
            'evidence': evidence,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment['decisions'].append(decision)
    
    def add_risk(self, risk: str, mitigation: str) -> None:
        """
        Log a potential risk and mitigation strategy
        
        Parameters:
        -----------
        risk : str
            Description of risk
        mitigation : str
            How to mitigate
        """
        risk_entry = {
            'risk': risk,
            'mitigation': mitigation,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment['risks'].append(risk_entry)
    
    def add_next_action(self, action: str, priority: str = 'medium') -> None:
        """
        Log next action for follow-up
        
        Parameters:
        -----------
        action : str
            Description of action
        priority : str
            'low', 'medium', or 'high'
        """
        action_entry = {
            'action': action,
            'priority': priority,
            'timestamp': datetime.now().isoformat()
        }
        self.experiment['next_actions'].append(action_entry)
    
    def save(self) -> Path:
        """
        Save experiment log to JSON file
        
        Returns:
        --------
        Path
            Path to saved log file
        """
        with open(self.experiment_file, 'w') as f:
            json.dump(self.experiment, f, indent=2)
        
        return self.experiment_file
    
    def load(self, log_file: str) -> Dict:
        """
        Load experiment log from file
        
        Parameters:
        -----------
        log_file : str
            Path to log file
            
        Returns:
        --------
        dict
            Experiment data
        """
        with open(log_file, 'r') as f:
            self.experiment = json.load(f)
        
        return self.experiment
    
    def summary(self) -> str:
        """
        Return formatted summary of experiment
        
        Returns:
        --------
        str
            Human-readable summary
        """
        summary = f"""
Experiment: {self.experiment_name}
Timestamp: {self.experiment['timestamp']}

=== Configuration ===
{json.dumps(self.experiment['config'], indent=2) if self.experiment['config'] else 'None'}

=== Results ===
{json.dumps(self.experiment['results'], indent=2) if self.experiment['results'] else 'None'}

=== Decisions ({len(self.experiment['decisions'])}) ===
"""
        for dec in self.experiment['decisions']:
            summary += f"- {dec['title']}\n  {dec['rationale']}\n"
        
        summary += f"\n=== Risks ({len(self.experiment['risks'])}) ===\n"
        for risk in self.experiment['risks']:
            summary += f"- {risk['risk']}\n  Mitigation: {risk['mitigation']}\n"
        
        summary += f"\n=== Next Actions ({len(self.experiment['next_actions'])}) ===\n"
        for action in self.experiment['next_actions']:
            summary += f"- [{action['priority'].upper()}] {action['action']}\n"
        
        return summary.strip()


def load_all_experiments(log_dir: str = "logs") -> Dict[str, Dict]:
    """
    Load all experiment logs from directory
    
    Parameters:
    -----------
    log_dir : str
        Directory containing experiment logs
        
    Returns:
    --------
    dict
        All experiments keyed by name
    """
    log_path = Path(log_dir)
    experiments = {}
    
    if not log_path.exists():
        return experiments
    
    for log_file in log_path.glob("*.json"):
        with open(log_file, 'r') as f:
            experiment = json.load(f)
            experiments[experiment['name']] = experiment
    
    return experiments
