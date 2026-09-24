The twin can stand in for real software: a DE-TEM-Channel SOAP server
(``de-twin serve --soap``) that DE-Server, de_microscope, de_autopilot and
de_ground_crew drive unchanged, a deapi-compatible fake DE-Server
(``--deapi``), and a shared-memory frame source that feeds a real DE-Server
through its processing pipeline (``--shm``, test pattern "External Frame Source
(Shared Memory)"). It can also follow a real or Dummy DE-TEM-Channel and a DENS
Impulse holder instead of simulating them.
