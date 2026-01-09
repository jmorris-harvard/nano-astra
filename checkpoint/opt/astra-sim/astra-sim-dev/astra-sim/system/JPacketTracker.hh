#ifndef __J_PACKET_TRACKER_HH__
#define __J_PACKET_TRACKER_HH__

#include <cstdint>
#include <string>
#include <vector>

namespace Jalil {

enum class PacketType {
  INVALID = 0,
  DATA,
  ACK,
  NACK
};

enum class PacketLocation {
  INVALID = 0,
  TRANSMIT,
  PROPAGATE,
  ARRIVE,
  RECEIVE
};

// Packet information
class PacketShadow {
  public:
    PacketShadow (
      PacketType type,
      PacketLocation loc,
      uint64_t node,
      uint64_t port, 
      uint64_t timestamp,
      uint64_t size
    );

    PacketType _type;
    PacketLocation _loc;
    uint64_t _node;
    uint64_t _port;
    uint64_t _timestamp;
    uint64_t _size;
};

// Logs all packets associated with attached data stream
class PacketTracker {
  public:
    PacketTracker (void);
    void addPacket (
      uint64_t type,
      uint64_t loc,
      uint64_t node,
      uint64_t port,
      uint64_t size
    );
    void addPacket (
      PacketType type,
      PacketLocation loc,
      uint64_t node,
      uint64_t port,
      uint64_t timestamp,
      uint64_t size
    );
    const std::vector<PacketShadow> &readPackets (void) const;

  private:
    std::vector<PacketShadow> _packets;
};

} // namespace Jalil

#endif /* __J_PACKET_TRACKER_HH__ */
