"""
Action history API endpoints.

This module provides endpoints for querying the action history
and audit trail of memory operations.

Example:
    GET /api/actions?limit=50&action_type=create
    GET /api/actions/memory/123
    GET /api/actions/stats
"""

from fastapi import APIRouter, Query, HTTPException
from typing import Optional, List, Dict, Any
from datetime import datetime

from memograph.core.action_logger import ActionLogger, ActionType

router = APIRouter(prefix="/actions", tags=["actions"])

# This will be set by the main app
action_logger: Optional[ActionLogger] = None


def set_action_logger(logger: ActionLogger):
    """Set the action logger instance."""
    global action_logger
    action_logger = logger


@router.get("")
async def get_actions(
    limit: int = Query(100, ge=1, le=1000, description="Maximum number of actions to return"),
    action_type: Optional[ActionType] = Query(None, description="Filter by action type"),
    memory_id: Optional[str] = Query(None, description="Filter by memory ID"),
    grouped: bool = Query(False, description="Group consecutive actions")
) -> Dict[str, Any]:
    """Get recent actions with optional filtering.
    
    Args:
        limit: Maximum number of actions
        action_type: Filter by action type
        memory_id: Filter by memory ID
        grouped: Whether to group consecutive actions
        
    Returns:
        Dictionary with actions list
    """
    if action_logger is None:
        raise HTTPException(status_code=500, detail="Action logger not initialized")
    
    actions = action_logger.get_recent_actions(
        limit=limit,
        action_type=action_type,
        memory_id=memory_id
    )
    
    if grouped:
        actions = action_logger.group_consecutive_actions(actions)
        return {
            "grouped": True,
            "groups": actions,
            "total_groups": len(actions)
        }
    
    return {
        "grouped": False,
        "actions": actions,
        "total": len(actions)
    }


@router.get("/memory/{memory_id}")
async def get_memory_history(memory_id: str) -> Dict[str, Any]:
    """Get all actions for a specific memory.
    
    Args:
        memory_id: Memory ID
        
    Returns:
        Dictionary with memory history
    """
    if action_logger is None:
        raise HTTPException(status_code=500, detail="Action logger not initialized")
    
    history = action_logger.get_memory_history(memory_id)
    
    return {
        "memory_id": memory_id,
        "actions": history,
        "total": len(history)
    }


@router.get("/stats")
async def get_action_stats() -> Dict[str, Any]:
    """Get statistics about logged actions.
    
    Returns:
        Dictionary with action statistics
    """
    if action_logger is None:
        raise HTTPException(status_code=500, detail="Action logger not initialized")
    
    stats = action_logger.get_stats()
    
    return stats


@router.delete("/clear")
async def clear_action_history(before_date: Optional[str] = Query(None, description="Clear actions before this date (ISO format)")) -> Dict[str, Any]:
    """Clear action history.
    
    Args:
        before_date: Optional ISO datetime to clear before
        
    Returns:
        Success message
    """
    if action_logger is None:
        raise HTTPException(status_code=500, detail="Action logger not initialized")
    
    if before_date:
        try:
            date_obj = datetime.fromisoformat(before_date)
            action_logger.clear_history(before_date=date_obj)
            return {"message": f"Cleared actions before {before_date}"}
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")
    else:
        action_logger.clear_history()
        return {"message": "Cleared all action history"}
