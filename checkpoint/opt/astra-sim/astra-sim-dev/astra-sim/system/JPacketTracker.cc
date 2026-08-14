#include "astra-sim/system/JPacketTracker.hh"

#include "astra-sim/system/Sys.hh"

// 9-23-2025 
#include <iostream>

using namespace std;
using namespace AstraSim;
using namespace Jalil;

PacketShadow::PacketShadow (
    PacketType type,
    PacketLocation loc,
    uint64_t node,
    uint64_t port,
    uint64_t timestamp,
    uint64_t size)
  : _type (type),
    _loc (loc),
    _node (node),
    _port (port),
    _timestamp (timestamp),
    _size (size) {}

// mirrors Sys::boostedTick()'s lookup of a live Sys instance; trace_enabled
// is set from the same config value on every rank, so any live instance
// reflects the simulation-wide setting.
static bool tracingEnabled (void) {
  for (auto &s : Sys::all_sys) {
    if (s != nullptr) {
      return s->trace_enabled;
    }
  }
  return false;
}

PacketTracker::PacketTracker (void)
  : _packets () {}

void PacketTracker::addPacket (
    PacketType type,
    PacketLocation loc,
    uint64_t node,
    uint64_t port,
    uint64_t timestamp,
    uint64_t size) {
  if (!tracingEnabled ()) {
    return;
  }
  _packets.emplace_back (type, loc, node, port, timestamp, size);
}

void PacketTracker::addPacket ( 
    uint64_t pType,
    uint64_t pLoc,
    uint64_t node,
    uint64_t port,
    uint64_t size) {
  PacketType type = static_cast<PacketType> (pType);
  PacketLocation loc = static_cast<PacketLocation> (pLoc);
  this->addPacket (type, loc, node, port, Sys::boostedTick (), size);
}

const vector<PacketShadow> &PacketTracker::readPackets (void) const {
  return _packets;
}
