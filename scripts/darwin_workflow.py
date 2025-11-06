#!/usr/bin/env python3
"""
🤖 Darwin Workflow Agent

Automatiza o workflow de desenvolvimento usando os agentes Darwin.
Coordena tarefas entre múltiplos repositórios e agentes.

Author: Dr. Demetrios Chiuratto Agourakis
Date: November 6, 2025
"""
import json
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional


class DarwinWorkflowAgent:
    """Agente para coordenar workflows usando Darwin"""
    
    def __init__(self, repo_path: Path):
        self.repo_path = Path(repo_path)
        self.sync_state_path = self.repo_path / "SYNC_STATE.json"
        self.agents_dir = self.repo_path / ".darwin" / "agents"
    
    def load_sync_state(self) -> Dict:
        """Carrega o estado de sincronização"""
        if self.sync_state_path.exists():
            with open(self.sync_state_path, 'r') as f:
                return json.load(f)
        return {
            "last_update": "",
            "active_agents": [],
            "locks": {},
            "last_actions": [],
            "version": "1.0.0",
            "repository": self.repo_path.name
        }
    
    def save_sync_state(self, state: Dict):
        """Salva o estado de sincronização"""
        state["last_update"] = datetime.now().isoformat()
        with open(self.sync_state_path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def run_omniscient_agent(self) -> Dict:
        """Executa o agente omnisciente"""
        print("🧠 Executando Darwin Omniscient Agent...")
        script = self.agents_dir / "darwin-omniscient-agent.sh"
        if script.exists():
            result = subprocess.run(
                [str(script)],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        return {"success": False, "error": "Script not found"}
    
    def run_sync_check(self) -> Dict:
        """Executa o sync check"""
        print("🔄 Executando Sync Check...")
        script = self.agents_dir / "sync-check.sh"
        if script.exists():
            result = subprocess.run(
                [str(script)],
                cwd=self.repo_path,
                capture_output=True,
                text=True
            )
            return {
                "success": result.returncode == 0,
                "output": result.stdout,
                "error": result.stderr
            }
        return {"success": False, "error": "Script not found"}
    
    def record_action(self, action: str, description: str, files_modified: List[str] = None):
        """Registra uma ação no SYNC_STATE"""
        state = self.load_sync_state()
        
        action_record = {
            "action": action,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "agent_id": "darwin_workflow_agent",
            "files_modified": files_modified or []
        }
        
        state["last_actions"].append(action_record)
        
        # Manter apenas últimas 20 ações
        if len(state["last_actions"]) > 20:
            state["last_actions"] = state["last_actions"][-20:]
        
        self.save_sync_state(state)
        print(f"✅ Ação registrada: {action}")
    
    def get_next_tasks(self) -> List[Dict]:
        """Obtém próximas tarefas baseado no estado do projeto"""
        tasks = []
        
        # Verificar STATUS_ATUAL.md para tarefas pendentes
        status_file = self.repo_path / "STATUS_ATUAL.md"
        if status_file.exists():
            content = status_file.read_text()
            if "PENDENTE" in content or "⏳" in content:
                tasks.append({
                    "type": "review",
                    "description": "Revisar STATUS_ATUAL.md para tarefas pendentes",
                    "priority": "medium"
                })
        
        # Verificar PENDING_ISSUES
        pending_file = self.repo_path / "docs" / "PENDING_ISSUES_PBPK_AND_CLEANUP.md"
        if pending_file.exists():
            tasks.append({
                "type": "development",
                "description": "Revisar issues pendentes de PBPK e cleanup",
                "priority": "high",
                "file": str(pending_file)
            })
        
        return tasks
    
    def workflow_start(self):
        """Inicia o workflow usando agentes Darwin"""
        print("=" * 80)
        print("🤖 DARWIN WORKFLOW AGENT")
        print("=" * 80)
        print()
        
        # 1. Executar omniscient agent
        omniscient_result = self.run_omniscient_agent()
        if omniscient_result["success"]:
            print("✅ Contexto carregado")
        else:
            print(f"⚠️  Erro ao carregar contexto: {omniscient_result.get('error')}")
        
        print()
        
        # 2. Executar sync check
        sync_result = self.run_sync_check()
        if sync_result["success"]:
            print("✅ Sincronização verificada")
        else:
            print(f"⚠️  Problemas de sincronização: {sync_result.get('error')}")
        
        print()
        
        # 3. Obter próximas tarefas
        tasks = self.get_next_tasks()
        if tasks:
            print("📋 Próximas tarefas identificadas:")
            for i, task in enumerate(tasks, 1):
                print(f"   {i}. [{task['priority'].upper()}] {task['description']}")
        
        print()
        
        # 4. Registrar ação
        self.record_action(
            "workflow_start",
            "Workflow iniciado usando Darwin agents",
            []
        )
        
        return {
            "omniscient": omniscient_result,
            "sync": sync_result,
            "tasks": tasks
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Darwin Workflow Agent")
    parser.add_argument(
        "--repo",
        type=str,
        default=".",
        help="Caminho do repositório (padrão: atual)"
    )
    
    args = parser.parse_args()
    
    agent = DarwinWorkflowAgent(args.repo)
    result = agent.workflow_start()
    
    print("=" * 80)
    print("✅ Workflow iniciado!")
    print("=" * 80)


if __name__ == "__main__":
    main()

