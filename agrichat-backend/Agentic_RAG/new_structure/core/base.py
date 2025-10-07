"""
Base classes and interfaces for AgriChat Agentic RAG system.

This module defines the core interfaces and abstract base classes that provide
structure and consistency across the agricultural AI system components.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum


class ResponseMode(Enum):
    """Enumeration of available response modes."""
    FAST = "fast"
    CREW_AI = "crew_ai"
    FALLBACK = "fallback"


class AgentRole(Enum):
    """Enumeration of agent roles in the system."""
    RETRIEVER = "retriever"
    GRADER = "grader"
    HALLUCINATION_CHECKER = "hallucination_checker"
    ANSWER_VALIDATOR = "answer_validator"


@dataclass
class QueryContext:
    """
    Context information for processing agricultural queries.
    
    Attributes:
        question: The user's agricultural question
        conversation_history: Previous Q&A pairs for context
        user_state: User's geographical state/region
        timestamp: When the query was made
        session_id: Unique session identifier
    """
    question: str
    conversation_history: Optional[List[Dict]] = None
    user_state: Optional[str] = None
    timestamp: Optional[str] = None
    session_id: Optional[str] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.conversation_history is None:
            self.conversation_history = []
        if self.user_state is None:
            self.user_state = "India"


@dataclass
class AgriResponse:
    """
    Standardized response structure for agricultural queries.
    
    Attributes:
        content: The main response content
        source: Source of the information (RAG, LLM, etc.)
        confidence: Confidence score (0.0 to 1.0)
        metadata: Additional metadata about the response
        processing_time: Time taken to generate response
        mode: Response mode used
    """
    content: str
    source: str
    confidence: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    mode: Optional[ResponseMode] = None
    
    def __post_init__(self):
        """Initialize default values after creation."""
        if self.metadata is None:
            self.metadata = {}


class BaseAgriTool(ABC):
    """
    Abstract base class for all agricultural tools.
    
    This class defines the interface that all agricultural processing tools
    must implement to ensure consistency and interoperability.
    """
    
    def __init__(self, name: str, description: str):
        """
        Initialize the agricultural tool.
        
        Args:
            name: Tool identifier
            description: Tool description and capabilities
        """
        self.name = name
        self.description = description
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the tool with required dependencies.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def process_query(self, context: QueryContext) -> AgriResponse:
        """
        Process an agricultural query and return a response.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            AgriResponse: Structured response with content and metadata
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if the tool is functioning correctly.
        
        Returns:
            bool: True if tool is healthy, False otherwise
        """
        pass
    
    def is_initialized(self) -> bool:
        """Check if tool has been properly initialized."""
        return self._initialized
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get tool capabilities and configuration.
        
        Returns:
            Dict containing tool capabilities and settings
        """
        return {
            "name": self.name,
            "description": self.description,
            "initialized": self._initialized,
            "healthy": self.health_check() if self._initialized else False
        }


class BaseAgriAgent(ABC):
    """
    Abstract base class for all agricultural AI agents.
    
    This class defines the interface for CrewAI agents that handle
    different aspects of agricultural query processing.
    """
    
    def __init__(self, role: AgentRole, agent_config: Dict[str, Any]):
        """
        Initialize the agricultural agent.
        
        Args:
            role: Agent role in the processing pipeline
            agent_config: Configuration parameters for the agent
        """
        self.role = role
        self.config = agent_config
        self._agent = None
    
    @abstractmethod
    def create_agent(self) -> Any:
        """
        Create and configure the CrewAI agent.
        
        Returns:
            Configured CrewAI Agent instance
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.
        
        Returns:
            List of capability descriptions
        """
        pass
    
    def get_agent(self) -> Any:
        """Get the underlying CrewAI agent instance."""
        if self._agent is None:
            self._agent = self.create_agent()
        return self._agent


class BaseResponseHandler(ABC):
    """
    Abstract base class for response handling strategies.
    
    This class defines the interface for different response handling
    approaches (Fast Mode, CrewAI Mode, etc.).
    """
    
    def __init__(self, mode: ResponseMode):
        """
        Initialize the response handler.
        
        Args:
            mode: Response handling mode
        """
        self.mode = mode
        self._initialized = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """
        Initialize the response handler with required components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_response(self, context: QueryContext) -> AgriResponse:
        """
        Generate response for agricultural query.
        
        Args:
            context: Query context with question and metadata
            
        Returns:
            AgriResponse: Generated response with metadata
        """
        pass
    
    @abstractmethod
    def can_handle(self, context: QueryContext) -> bool:
        """
        Check if this handler can process the given query.
        
        Args:
            context: Query context to evaluate
            
        Returns:
            bool: True if handler can process query, False otherwise
        """
        pass
    
    def is_available(self) -> bool:
        """Check if response handler is available and initialized."""
        return self._initialized and self.health_check()
    
    @abstractmethod
    def health_check(self) -> bool:
        """
        Check if response handler is functioning correctly.
        
        Returns:
            bool: True if handler is healthy, False otherwise
        """
        pass


class AgriSystemInterface(ABC):
    """
    Main interface for the agricultural AI system.
    
    This class defines the primary interface that external systems
    (like FastAPI backend) use to interact with the AgriChat system.
    """
    
    @abstractmethod
    def process_question(self, 
                        question: str, 
                        conversation_history: Optional[List[Dict]] = None,
                        user_state: Optional[str] = None) -> str:
        """
        Process an agricultural question and return response.
        
        Args:
            question: User's agricultural question
            conversation_history: Previous conversation context
            user_state: User's geographical state/region
            
        Returns:
            str: Generated response to the agricultural question
        """
        pass
    
    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get comprehensive system status and health information.
        
        Returns:
            Dict containing system status and component health
        """
        pass
    
    @abstractmethod
    def initialize_system(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if system initialized successfully, False otherwise
        """
        pass
